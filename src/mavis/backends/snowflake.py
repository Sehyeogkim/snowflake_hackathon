from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..config import require_env
from ..labels import UNSAFE_LABELS, normalize_label
from ..models import CandidateSet, ClipRecord, InferenceResult
from ..prompts import safety_prompt
from .base import InferenceBackend

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _json_from_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"AI_COMPLETE returned non-JSON output: {text[:300]}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("AI_COMPLETE JSON output must be an object")
    return parsed


class SnowflakeInferenceBackend(InferenceBackend):
    """Snowflake Cortex image inference with one connection per worker thread."""

    evidence_source = "snowflake"

    def __init__(
        self,
        labels: tuple[str, ...],
        cheap_model: str,
        strong_model: str,
        query_tag_prefix: str = "mavis|seed",
    ) -> None:
        required = require_env(
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
            "SNOWFLAKE_IMAGE_STAGE",
        )
        self.labels = labels
        self.cheap_model = cheap_model
        self.strong_model = strong_model
        self.query_tag_prefix = query_tag_prefix
        self.stage = required["SNOWFLAKE_IMAGE_STAGE"]
        if not _IDENTIFIER.fullmatch(self.stage):
            raise ValueError("SNOWFLAKE_IMAGE_STAGE must be an unquoted simple identifier")
        self._local = threading.local()
        self._staged_by_thread = threading.local()
        self._connection_args: dict[str, Any] = {
            "account": required["SNOWFLAKE_ACCOUNT"],
            "user": required["SNOWFLAKE_USER"],
            "warehouse": required["SNOWFLAKE_WAREHOUSE"],
            "database": required["SNOWFLAKE_DATABASE"],
            "schema": required["SNOWFLAKE_SCHEMA"],
            "role": os.getenv("SNOWFLAKE_ROLE") or None,
            "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            "session_parameters": {"QUERY_TAG": query_tag_prefix},
        }
        if os.getenv("SNOWFLAKE_PASSWORD"):
            self._connection_args["password"] = os.environ["SNOWFLAKE_PASSWORD"]
        if os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"):
            self._connection_args["private_key_file"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]

    def _connection(self):
        connection = getattr(self._local, "connection", None)
        if connection is None or connection.is_closed():
            try:
                import snowflake.connector
            except ImportError as exc:
                raise RuntimeError(
                    "Install Snowflake support first: pip install -e '.[snowflake]'"
                ) from exc
            args = {key: value for key, value in self._connection_args.items() if value is not None}
            connection = snowflake.connector.connect(**args)
            self._local.connection = connection
        return connection

    def _stage_files(self, cursor, clip: ClipRecord, action: str, files: list[Path]) -> list[str]:
        del action
        staged = getattr(self._staged_by_thread, "paths", None)
        if staged is None:
            staged = set()
            self._staged_by_thread.paths = staged
        relative_paths: list[str] = []
        for path in files:
            relative = f"{clip.clip_id}/{path.name}"
            if relative not in staged:
                uri = path.resolve().as_posix().replace("'", "''")
                stage_target = f"@{self.stage}/{clip.clip_id}/"
                cursor.execute(
                    f"PUT 'file://{uri}' {stage_target} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                )
                staged.add(relative)
            relative_paths.append(relative)
        return relative_paths

    def infer(self, clip: ClipRecord, candidates: CandidateSet, action: str) -> InferenceResult:
        started = time.perf_counter()
        model = self.cheap_model if action == "cheap_single" else self.strong_model
        files = candidates.for_action(action)
        cursor = self._connection().cursor()
        query_id: str | None = None
        try:
            safe_tag = f"{self.query_tag_prefix}|{clip.clip_id}|{action}".replace("'", "''")
            cursor.execute(f"ALTER SESSION SET QUERY_TAG = '{safe_tag}'")
            relative_paths = self._stage_files(cursor, clip, action, files)
            prompt = safety_prompt(self.labels, action, len(files))

            placeholders = " and ".join(f"image {{{index}}}" for index in range(len(files)))
            prompt_template = f"{prompt}\nObservations: {placeholders}."
            escaped_paths = [path.replace("'", "''") for path in relative_paths]
            file_sql = ", ".join(f"TO_FILE('@{self.stage}', '{path}')" for path in escaped_paths)
            sql = f"SELECT AI_COMPLETE(%s, PROMPT(%s, {file_sql}))"
            cursor.execute(sql, (model, prompt_template))
            query_id = cursor.sfqid
            payload = _json_from_response(cursor.fetchone()[0])
            prediction = normalize_label(str(payload.get("prediction", "")))
            if prediction not in self.labels:
                raise ValueError(f"Model returned an unsupported label: {prediction}")
            raw_scores = payload.get("scores") or {}
            scores = {
                label: float(raw_scores.get(label, raw_scores.get(label.replace("_", " "), 0.0)))
                for label in self.labels
            }
            score_risk = sum(scores.get(label, 0.0) for label in UNSAFE_LABELS)
            reported_risk = max(0.0, min(1.0, float(payload.get("risk", 0.0))))
            return InferenceResult(
                action=action,
                model=model,
                prediction=prediction,
                scores=scores,
                scene=str(payload.get("scene", "factory safety scene")),
                # Never let a prose risk field understate the model's own unsafe
                # class mass. This closes an inconsistent-output acceptance path.
                risk=max(reported_risk, min(1.0, score_risk)),
                need_temporal_context=bool(payload.get("need_temporal_context", False)),
                query_id=query_id,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response=payload,
            )
        except Exception as exc:
            return InferenceResult(
                action=action,
                model=model,
                prediction="",
                scores={},
                scene="",
                risk=0.0,
                need_temporal_context=False,
                query_id=query_id or getattr(cursor, "sfqid", None),
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response={},
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            cursor.close()

    def reconcile_costs(self, query_ids: list[str], wait_seconds: int = 0) -> dict[str, float]:
        if wait_seconds:
            time.sleep(wait_seconds)
        ids = sorted({query_id for query_id in query_ids if query_id})
        if not ids:
            return {}
        cursor = self._connection().cursor()
        try:
            placeholders = ",".join(["%s"] * len(ids))
            cursor.execute(
                "SELECT QUERY_ID, SUM(CREDITS) "
                "FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AI_FUNCTIONS_USAGE_HISTORY "
                f"WHERE QUERY_ID IN ({placeholders}) GROUP BY QUERY_ID",
                ids,
            )
            return {str(query_id): float(credits) for query_id, credits in cursor.fetchall()}
        finally:
            cursor.close()

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None and not connection.is_closed():
            connection.close()
