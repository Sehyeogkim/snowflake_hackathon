from __future__ import annotations

import base64
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from ..labels import UNSAFE_LABELS, normalize_label
from ..models import CandidateSet, ClipRecord, InferenceResult
from ..prompts import safety_prompt
from .base import InferenceBackend

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_TERRA_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_STRONG_ACTIONS = frozenset({"strong_single", "strong_multi", "crop_strong"})
_IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _image_data_url(path: Path) -> str:
    media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.casefold())
    if media_type is None:
        supported = ", ".join(sorted(_IMAGE_MEDIA_TYPES))
        raise ValueError(f"Unsupported image type for {path.name}; expected one of: {supported}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _output_text(response_payload: dict[str, Any]) -> str:
    """Extract text from the raw HTTP form of a Responses API response."""
    for item in response_payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("OpenAI response did not contain output_text")


def _response_schema(labels: tuple[str, ...]) -> dict[str, Any]:
    score_properties = {
        label: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for label in labels
    }
    return {
        "type": "object",
        "properties": {
            "prediction": {"type": "string", "enum": list(labels)},
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": list(labels),
                "additionalProperties": False,
            },
            "scene": {"type": "string"},
            "risk": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "need_temporal_context": {"type": "boolean"},
        },
        "required": [
            "prediction",
            "scores",
            "scene",
            "risk",
            "need_temporal_context",
        ],
        "additionalProperties": False,
    }


class OpenAIInferenceBackend(InferenceBackend):
    """OpenAI Responses API vision backend with strict structured output."""

    evidence_source = "openai"

    def __init__(
        self,
        labels: tuple[str, ...],
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_seconds: float = 60.0,
        image_detail: str = "low",
        max_attempts: int = 3,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 4.0,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
    ) -> None:
        if not labels:
            raise ValueError("At least one label is required")
        if len(set(labels)) != len(labels):
            raise ValueError("Labels must be unique")
        if image_detail not in {"auto", "low", "high", "original"}:
            raise ValueError("image_detail must be auto, low, high, or original")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_base_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("Retry delays cannot be negative")

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI inference")

        self.labels = labels
        self.model = model
        self.image_detail = image_detail
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._sleep = sleep
        self._responses_url = f"{base_url.rstrip('/')}/responses"
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
        self._authorization = f"Bearer {resolved_key}" if client is not None else None

    def _request_payload(self, paths: Sequence[Path], action: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": safety_prompt(self.labels, action, len(paths)),
            }
        ]
        content.extend(
            {
                "type": "input_image",
                "image_url": _image_data_url(path),
                "detail": self.image_detail,
            }
            for path in paths
        )
        return {
            "model": self.model,
            "store": False,
            "reasoning": {"effort": "none"},
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "mavis_safety_inference",
                    "strict": True,
                    "schema": _response_schema(self.labels),
                }
            },
        }

    def _post_with_retry(self, request_payload: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": self._authorization} if self._authorization else None
        response: httpx.Response | None = None
        for attempt in range(self.max_attempts):
            response = self._client.post(
                self._responses_url,
                json=request_payload,
                headers=headers,
            )
            retryable = response.status_code == 429 or 500 <= response.status_code <= 599
            if not retryable or attempt + 1 >= self.max_attempts:
                return response
            delay = min(self.retry_base_seconds * (2**attempt), self.retry_max_seconds)
            self._sleep(delay)
        raise RuntimeError("OpenAI retry loop ended without a response")

    def infer_paths(
        self,
        paths: Sequence[str | Path],
        action: str,
    ) -> InferenceResult:
        """Infer over chronological image paths in one Responses API request."""
        started = time.perf_counter()
        query_id: str | None = None
        raw_response: dict[str, Any] = {}
        try:
            ordered_paths = tuple(Path(path) for path in paths)
            if not ordered_paths:
                raise ValueError("At least one chronological image path is required")
            request_payload = self._request_payload(ordered_paths, action)
            response = self._post_with_retry(request_payload)
            response.raise_for_status()
            decoded = response.json()
            if not isinstance(decoded, dict):
                raise ValueError("OpenAI response must be a JSON object")
            raw_response = dict(decoded)
            query_id_value = decoded.get("id")
            query_id = str(query_id_value) if query_id_value is not None else None

            parsed = json.loads(_output_text(decoded))
            if not isinstance(parsed, dict):
                raise ValueError("OpenAI structured output must be a JSON object")
            raw_response["parsed"] = parsed

            prediction = normalize_label(str(parsed.get("prediction", "")))
            if prediction not in self.labels:
                raise ValueError(f"Model returned an unsupported label: {prediction}")
            raw_scores = parsed.get("scores")
            if not isinstance(raw_scores, dict):
                raise ValueError("Model response scores must be an object")
            scores = {label: float(raw_scores[label]) for label in self.labels}
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in scores.values()):
                raise ValueError("Model response scores must be finite values between 0 and 1")

            reported_risk = float(parsed.get("risk", 0.0))
            if not math.isfinite(reported_risk):
                raise ValueError("Model response risk must be finite")
            reported_risk = max(0.0, min(1.0, reported_risk))
            score_risk = sum(scores.get(label, 0.0) for label in UNSAFE_LABELS)

            return InferenceResult(
                action=action,
                model=self.model,
                prediction=prediction,
                scores=scores,
                scene=str(parsed.get("scene", "factory safety scene")),
                risk=max(reported_risk, min(1.0, score_risk)),
                need_temporal_context=bool(parsed.get("need_temporal_context", False)),
                query_id=query_id,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response=raw_response,
            )
        except httpx.HTTPStatusError as exc:
            error = f"HTTPStatusError: OpenAI Responses API returned HTTP {exc.response.status_code}"
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: OpenAI Responses API request failed"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        return InferenceResult(
            action=action,
            model=self.model,
            prediction="",
            scores={},
            scene="",
            risk=0.0,
            need_temporal_context=False,
            query_id=query_id,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw_response,
            error=error,
        )

    def infer(self, clip: ClipRecord, candidates: CandidateSet, action: str) -> InferenceResult:
        del clip
        try:
            paths = candidates.for_action(action)
        except Exception as exc:
            return InferenceResult(
                action=action,
                model=self.model,
                prediction="",
                scores={},
                scene="",
                risk=0.0,
                need_temporal_context=False,
                query_id=None,
                latency_ms=0.0,
                raw_response={},
                error=f"{type(exc).__name__}: {exc}",
            )
        return self.infer_paths(paths, action)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class TieredOpenAIInferenceBackend(InferenceBackend):
    """Route cheap and strong actions to Luna/low and Terra/original respectively."""

    evidence_source = "openai"

    def __init__(
        self,
        labels: tuple[str, ...],
        api_key: str | None = None,
        cheap_model: str = DEFAULT_OPENAI_MODEL,
        strong_model: str = DEFAULT_OPENAI_TERRA_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 4.0,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
    ) -> None:
        self.cheap_model = cheap_model
        self.strong_model = strong_model
        shared: dict[str, Any] = {
            "labels": labels,
            "api_key": api_key,
            "base_url": base_url,
            "timeout_seconds": timeout_seconds,
            "max_attempts": max_attempts,
            "retry_base_seconds": retry_base_seconds,
            "retry_max_seconds": retry_max_seconds,
            "sleep": sleep,
            "client": client,
        }
        self.cheap_backend = OpenAIInferenceBackend(
            model=cheap_model,
            image_detail="low",
            **shared,
        )
        self.strong_backend = OpenAIInferenceBackend(
            model=strong_model,
            image_detail="original",
            **shared,
        )

    def _backend(self, action: str, tier: str | None = None) -> OpenAIInferenceBackend:
        if tier is not None:
            normalized_tier = tier.casefold()
            if normalized_tier in {"cheap", "luna"}:
                return self.cheap_backend
            if normalized_tier in {"strong", "terra"}:
                return self.strong_backend
            raise ValueError("tier must be cheap/luna or strong/terra")
        if action == "cheap_single":
            return self.cheap_backend
        if action in _STRONG_ACTIONS:
            return self.strong_backend
        raise ValueError(f"Unsupported tiered OpenAI action: {action}")

    def _routing_error(self, action: str, exc: Exception) -> InferenceResult:
        return InferenceResult(
            action=action,
            model="",
            prediction="",
            scores={},
            scene="",
            risk=0.0,
            need_temporal_context=False,
            query_id=None,
            latency_ms=0.0,
            raw_response={},
            error=f"{type(exc).__name__}: {exc}",
        )

    def infer(self, clip: ClipRecord, candidates: CandidateSet, action: str) -> InferenceResult:
        try:
            backend = self._backend(action)
        except ValueError as exc:
            return self._routing_error(action, exc)
        return backend.infer(clip, candidates, action)

    def infer_paths(
        self,
        paths: Sequence[str | Path],
        action: str,
        *,
        tier: str | None = None,
    ) -> InferenceResult:
        """Infer arbitrary chronological paths, optionally selecting a tier explicitly."""
        try:
            backend = self._backend(action, tier)
        except ValueError as exc:
            return self._routing_error(action, exc)
        return backend.infer_paths(paths, action)

    def close(self) -> None:
        self.cheap_backend.close()
        self.strong_backend.close()
