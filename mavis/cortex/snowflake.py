"""The real Cortex client: AI_CLASSIFY and AI_COMPLETE over Snowflake.

Frames are JPEG-encoded, PUT to an internal stage, and referenced with
``TO_FILE(@stage, path)`` — the FILE type is how Cortex AI functions take image
input. ``AI_COMPLETE`` is called with ``show_details => TRUE`` so that the
returned object carries per-call token usage, which is what the scheduler
charges against. Snowflake's own ``CORTEX_AI_FUNCTIONS_USAGE_HISTORY`` view is
the authoritative record and is reconciled separately (see :mod:`mavis.usage`),
because that view lags real time by up to a few hours.

Not yet exercised against a live account — there is no Snowflake account on this
machine at time of writing. Everything here is written against the documented
SQL surface and is expected to need a smoke test (``mavis smoke``) before the
benchmark is trusted.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..types import CostRecord, InferenceResult, SceneSummary

#: Model names for the two tiers. Both must be multimodal.
CHEAP_MODEL = os.environ.get("MAVIS_CHEAP_MODEL", "claude-haiku-4-5")
STRONG_MODEL = os.environ.get("MAVIS_STRONG_MODEL", "claude-sonnet-5")


@dataclass(slots=True)
class SnowflakeSettings:
    account: str
    user: str
    warehouse: str
    database: str
    schema: str
    role: str | None = None
    password: str | None = None
    private_key_path: str | None = None
    stage: str = "MAVIS_FRAMES"

    @classmethod
    def from_env(cls) -> "SnowflakeSettings":
        missing = [
            k
            for k in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE")
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(
                "missing Snowflake environment variables: "
                + ", ".join(missing)
                + "\nSet them in .env (see .env.example) or run with --cortex mock."
            )
        return cls(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ.get("SNOWFLAKE_DATABASE", "MAVIS"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            private_key_path=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"),
            stage=os.environ.get("MAVIS_STAGE", "MAVIS_FRAMES"),
        )


class SnowflakeCortex:
    """CortexClient backed by a live Snowflake account."""

    name = "snowflake"
    estimated = False
    cost_unit = "credits"

    def __init__(
        self,
        settings: SnowflakeSettings | None = None,
        *,
        jpeg_quality: int = 80,
        max_edge: int = 768,
        scratch: str | Path = "data/frames",
    ):
        import snowflake.connector  # imported lazily: optional dependency

        self.settings = settings or SnowflakeSettings.from_env()
        self.jpeg_quality = jpeg_quality
        self.max_edge = max_edge
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "account": self.settings.account,
            "user": self.settings.user,
            "warehouse": self.settings.warehouse,
            "database": self.settings.database,
            "schema": self.settings.schema,
        }
        if self.settings.role:
            kwargs["role"] = self.settings.role
        if self.settings.password:
            kwargs["password"] = self.settings.password
        if self.settings.private_key_path:
            kwargs["private_key_file"] = self.settings.private_key_path

        self.conn = snowflake.connector.connect(**kwargs)
        self._ensure_stage()

    # -- setup -------------------------------------------------------------

    def _ensure_stage(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"CREATE STAGE IF NOT EXISTS {self.settings.stage} "
                "DIRECTORY = (ENABLE = TRUE) "
                "ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
            )

    # -- frame upload ------------------------------------------------------

    def _encode(self, image: np.ndarray) -> Path:
        h, w = image.shape[:2]
        scale = min(self.max_edge / max(h, w), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        path = self.scratch / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return path

    def _stage(self, image: np.ndarray) -> str:
        """Encode and PUT one frame, returning its stage-relative path."""
        local = self._encode(image)
        with self.conn.cursor() as cur:
            cur.execute(
                f"PUT file://{local.as_posix()} @{self.settings.stage}/frames/ "
                "AUTO_COMPRESS = FALSE OVERWRITE = TRUE"
            )
        local.unlink(missing_ok=True)
        return f"frames/{local.name}"

    def _file_expr(self, stage_path: str) -> str:
        return f"TO_FILE('@{self.settings.stage}', '{stage_path}')"

    # -- CortexClient ------------------------------------------------------

    def classify(self, image: np.ndarray, categories: Sequence[str]):
        stage_path = self._stage(image)
        cats = json.dumps(list(categories))
        sql = f"""
            SELECT AI_CLASSIFY({self._file_expr(stage_path)}, PARSE_JSON(%s)) AS result
        """
        started = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute(sql, (cats,))
            row = cur.fetchone()
            query_id = cur.sfqid
        latency = time.perf_counter() - started

        payload = _as_json(row[0]) if row else {}
        labels = payload.get("labels") or []
        # AI_CLASSIFY returns the chosen category; risk is derived from which
        # category won, since the function itself does not emit a score.
        risk = _category_risk(labels)
        return (
            SceneSummary(labels=labels, objects=[], risk=risk, raw=payload),
            CostRecord(
                model="ai_classify",
                prompt_tokens=0,  # AI_CLASSIFY does not expose token detail
                completion_tokens=0,
                credits=None,  # reconciled from usage history
                query_id=query_id,
                latency_s=latency,
            ),
        )

    def complete(self, images: Sequence[np.ndarray], prompt: str, *, strong: bool):
        model = STRONG_MODEL if strong else CHEAP_MODEL
        stage_paths = [self._stage(im) for im in images]
        files = ", ".join(self._file_expr(p) for p in stage_paths)
        sql = f"""
            SELECT AI_COMPLETE(
                model => %s,
                prompt => PROMPT(%s, {files}),
                show_details => TRUE
            ) AS result
        """
        placeholders = " ".join(f"{{{i}}}" for i in range(len(stage_paths)))
        started = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute(sql, (model, f"{placeholders}\n\n{prompt}"))
            row = cur.fetchone()
            query_id = cur.sfqid
        latency = time.perf_counter() - started

        payload = _as_json(row[0]) if row else {}
        usage = payload.get("usage", {})
        text = _first_text(payload)
        parsed = _parse_hazard_json(text)

        return (
            InferenceResult(
                hazard_prob=parsed["hazard_prob"],
                confidence=parsed["confidence"],
                rationale=parsed["rationale"],
                evidence=parsed["evidence"],
                raw=payload,
            ),
            CostRecord(
                model=model,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                credits=None,  # reconciled from usage history
                query_id=query_id,
                latency_s=latency,
            ),
        )

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover - best-effort teardown
            pass


# -- response parsing ------------------------------------------------------


def _as_json(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value if isinstance(value, str) else value.decode("utf-8", "replace")}
    return {}


def _first_text(payload: dict) -> str:
    """Pull the assistant text out of an AI_COMPLETE show_details envelope."""
    if "choices" in payload:
        choices = payload["choices"]
        if choices:
            first = choices[0]
            return first.get("messages") or first.get("message") or first.get("text") or ""
    for key in ("messages", "message", "text", "raw"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    return ""


def _parse_hazard_json(text: str) -> dict:
    """Best-effort extraction of the hazard verdict.

    A VLM that ignores the JSON instruction must not crash a benchmark run, so a
    failure degrades to an explicitly low-confidence neutral verdict rather than
    raising. Those show up in the trace and can be counted.
    """
    fallback = {
        "hazard_prob": 0.5,
        "confidence": 0.05,
        "rationale": "unparseable model response",
        "evidence": [],
    }
    if not text:
        return fallback
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return fallback
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return fallback
    return {
        "hazard_prob": float(obj.get("hazard_prob", 0.5)),
        "confidence": float(obj.get("confidence", 0.3)),
        "rationale": str(obj.get("rationale", "")),
        "evidence": list(obj.get("evidence", []) or []),
    }


def _category_risk(labels: list[str]) -> float:
    """Map an AI_CLASSIFY category back to a coarse risk score.

    The cheap tier is intentionally blunt: it says what the scene *is*, and the
    unsafe/safe variants of the same activity land close together. That gap is
    what makes a strong call worth paying for.
    """
    from ..dataset import CLASSIFY_CATEGORIES

    unsafe_markers = ("outside marked", "reaching into", "cover open", "oversized")
    for label in labels:
        low = label.lower()
        if any(m in low for m in unsafe_markers):
            return 0.66
        if "no person" in low:
            return 0.05
    return 0.33 if labels else 0.2
