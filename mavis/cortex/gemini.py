"""Gemini-backed Cortex client.

Same two-tier structure as the Snowflake path — a cheap model for "what is this
scene" and a capable one for "is this actually dangerous" — so the scheduler is
unchanged. What differs is how cost is established.

**Tokens are measured; money is derived.** Every response carries
``usageMetadata`` with exact token counts, which is what the scheduler charges
against and what the benchmark reports as its primary cost basis. Converting
those to dollars needs a published price list, and price lists change, so
:data:`PRICING` is dated, sourced, and overridable — and an unpriced model raises
rather than silently costing nothing.

Two details that are easy to get wrong and expensive if you do:

* ``thoughtsTokenCount`` is billed as output but is *not* included in
  ``candidatesTokenCount``. Ignoring it undercounts a thinking model's cost
  several-fold. :func:`_usage` folds it into completion tokens.
* Image tokens dominate everything else — a 320×240 frame costs ~1,000 prompt
  tokens against ~34 for the prompt text. Cost is therefore essentially per
  image, which makes frame resolution a direct cost lever (``max_edge``) and
  makes skipping a frame the single biggest saving available.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from ..types import CostRecord, InferenceResult, SceneSummary

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

#: Published list prices in USD per 1M tokens, as (input, output).
#: Source: ai.google.dev/gemini-api/docs/pricing, checked 2026-08-07.
#: Verify before quoting a dollar figure — rates change and preview models
#: change more often. Token counts in the benchmark are measured, not derived,
#: so a stale rate here affects only the USD column.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.1-pro-preview": (2.00, 18.00),
    # gemini-2.5-pro is priced (1.25, 10.00) but the API rejects it for accounts
    # that had not already used it, so it is not offered as a default.
}

#: Verified working against this key on 2026-08-07. The tiers are further apart
#: than the price list alone suggests: the 3.x image tokenizer charges ~1,100
#: prompt tokens for a 640×360 frame where 2.5-flash-lite charges ~270, so the
#: strong tier costs more per token *and* uses more of them per frame.
CHEAP_MODEL = os.environ.get("MAVIS_CHEAP_MODEL", "gemini-2.5-flash-lite")
STRONG_MODEL = os.environ.get("MAVIS_STRONG_MODEL", "gemini-3.1-pro-preview")

#: Schema forced on every hazard verdict, so parsing cannot be the failure mode.
HAZARD_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hazard_prob": {"type": "NUMBER"},
        "confidence": {"type": "NUMBER"},
        "rationale": {"type": "STRING"},
        "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["hazard_prob", "confidence", "rationale"],
}


class UnpricedModel(RuntimeError):
    """Raised rather than silently reporting a model as free."""


@dataclass(slots=True)
class _Response:
    text: str
    usage: dict[str, Any]
    latency_s: float


class GeminiCortex:
    """CortexClient backed by the Gemini API."""

    name = "gemini"
    estimated = False
    cost_unit = "USD"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cheap_model: str | None = None,
        strong_model: str | None = None,
        jpeg_quality: int = 80,
        max_edge: int = 768,
        timeout: float = 120.0,
        max_retries: int = 4,
        cheap_thinking: bool = False,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in .env (see .env.example) "
                "or run with --cortex mock."
            )
        self.cheap_model = cheap_model or CHEAP_MODEL
        self.strong_model = strong_model or STRONG_MODEL
        self.jpeg_quality = jpeg_quality
        self.max_edge = max_edge
        self.timeout = timeout
        self.max_retries = max_retries
        # Thinking tokens bill as output. Leaving them on for the cheap tier
        # would erase the price gap the whole scheduler trades on.
        self.cheap_thinking = cheap_thinking

        for model in (self.cheap_model, self.strong_model):
            if model not in PRICING:
                raise UnpricedModel(
                    f"no published rate for {model!r}. Add it to mavis.cortex.gemini.PRICING "
                    "with a source, or pick a priced model — reporting a cost of zero "
                    "would be worse than failing here."
                )

    # -- CortexClient ------------------------------------------------------

    def classify(self, image: np.ndarray, categories: Sequence[str]):
        """Cheap scene understanding: pick a category and rate the risk."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "enum": list(categories)},
                "objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                "risk": {"type": "NUMBER"},
            },
            "required": ["category", "objects", "risk"],
        }
        prompt = (
            "Classify this factory CCTV frame into exactly one category. "
            "List the salient objects you can see, and give a rough risk score "
            "from 0 (clearly safe) to 1 (clearly dangerous). Describe only what "
            "is visible; do not speculate."
        )
        resp = self._generate(self.cheap_model, [image], prompt, schema, thinking=self.cheap_thinking)
        payload = _loads(resp.text)
        return (
            SceneSummary(
                labels=[payload["category"]] if payload.get("category") else [],
                objects=[str(o) for o in payload.get("objects", [])][:6],
                risk=_clamp(payload.get("risk", 0.3)),
                raw=payload,
            ),
            self._cost(self.cheap_model, resp),
        )

    def complete(self, images: Sequence[np.ndarray], prompt: str, *, strong: bool):
        """VLM reasoning over one or more frames."""
        model = self.strong_model if strong else self.cheap_model
        resp = self._generate(model, list(images), prompt, HAZARD_SCHEMA, thinking=strong)
        payload = _loads(resp.text)
        return (
            InferenceResult(
                hazard_prob=_clamp(payload.get("hazard_prob", 0.5)),
                # An unparseable answer must not masquerade as a confident one.
                confidence=_clamp(payload.get("confidence", 0.05 if not payload else 0.5)),
                rationale=str(payload.get("rationale", "")),
                evidence=[str(e) for e in payload.get("evidence", []) or []],
                raw=payload,
            ),
            self._cost(model, resp),
        )

    def close(self) -> None:  # pragma: no cover - stateless HTTP
        pass

    # -- HTTP --------------------------------------------------------------

    def _generate(
        self,
        model: str,
        images: list[np.ndarray],
        prompt: str,
        schema: dict,
        *,
        thinking: bool,
    ) -> _Response:
        parts: list[dict] = [
            {"inline_data": {"mime_type": "image/jpeg", "data": self._encode(im)}}
            for im in images
        ]
        if len(images) > 1:
            parts.append({"text": "The frames above are in chronological order."})
        parts.append({"text": prompt})

        config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.0,  # a benchmark has to be reproducible
            "maxOutputTokens": 2048,
        }
        if not thinking:
            config["thinkingConfig"] = {"thinkingBudget": 0}

        body = json.dumps({"contents": [{"parts": parts}], "generationConfig": config}).encode()
        url = f"{API_ROOT}/{model}:generateContent"

        started = time.perf_counter()
        payload = self._post(url, body)
        latency = time.perf_counter() - started

        candidates = payload.get("candidates") or []
        text = ""
        if candidates:
            for part in candidates[0].get("content", {}).get("parts", []):
                text += part.get("text", "")
        return _Response(text=text, usage=payload.get("usageMetadata", {}), latency_s=latency)

    def _post(self, url: str, body: bytes) -> dict:
        """POST with backoff on rate limits and transient server errors."""
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf-8", "replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 30))
                    last = exc
                    continue
                raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 30))
                    last = exc
                    continue
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
        raise RuntimeError(f"Gemini request failed after {self.max_retries} attempts: {last}")

    # -- encoding and cost -------------------------------------------------

    def _encode(self, image: np.ndarray) -> str:
        h, w = image.shape[:2]
        scale = min(self.max_edge / max(h, w), 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame")
        return base64.b64encode(buf).decode("ascii")

    def _cost(self, model: str, resp: _Response) -> CostRecord:
        prompt_tokens, completion_tokens = _usage(resp.usage)
        rate_in, rate_out = PRICING[model]
        usd = prompt_tokens / 1e6 * rate_in + completion_tokens / 1e6 * rate_out
        return CostRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            credits=usd,
            query_id=None,
            latency_s=resp.latency_s,
            estimated=False,
        )


# -- helpers ---------------------------------------------------------------


def _usage(usage: dict) -> tuple[int, int]:
    """Extract billable prompt and completion tokens.

    ``thoughtsTokenCount`` is billed at the output rate but sits outside
    ``candidatesTokenCount``, so it has to be added explicitly.
    """
    prompt = int(usage.get("promptTokenCount", 0))
    completion = int(usage.get("candidatesTokenCount", 0)) + int(
        usage.get("thoughtsTokenCount", 0)
    )
    return prompt, completion


def _loads(text: str) -> dict:
    """Parse a response body, returning {} rather than raising on garbage."""
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo
