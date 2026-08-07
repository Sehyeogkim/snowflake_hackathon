from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
from google import genai
from google.genai import types

from .labels import CANONICAL_LABELS, UNSAFE_LABELS
from .models import ClipRecord

CHEAP_MODEL = "gemini-3.1-flash-lite"
STRONG_MODEL = "gemini-3.1-pro-preview"
PROMPT_VERSION = "mavis-dense-teacher-v2"
# Kept stable because it is part of every delivered Experience ID.
EXPERIENCE_VERSION = "mavis-dense-teacher-v3"
MAX_OUTPUT_TOKENS = 1024


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


MODEL_PRICES = {
    CHEAP_MODEL: ModelPrice(0.25, 1.50),
    STRONG_MODEL: ModelPrice(2.00, 12.00),
}

CLASS_GUIDE = """
- safe_walkway: a pedestrian remains within the designated safe walkway.
- safe_walkway_violation: a pedestrian leaves or crosses outside the walkway unsafely.
- authorized_intervention: an authorized worker safely operates or intervenes on equipment.
- unauthorized_intervention: a person performs an unauthorized or unsafe intervention.
- closed_panel_cover: the equipment panel or cover is closed.
- opened_panel_cover: the equipment panel or cover is left open or is opened.
- safe_carrying: a load is carried safely by a person or vehicle.
- carrying_overload_with_forklift: a forklift carries an overloaded or unsafe load.
""".strip()

BASE_PROMPT = f"""
Analyze this real factory-safety video sampled at 10 frames per second. Classify exactly one
behavior from the fixed taxonomy below. Use temporal evidence from the video itself. Do not
infer a class from filenames or metadata.

{CLASS_GUIDE}

Return probabilities for all eight classes that sum to approximately 1. Identify up to three
short critical time windows (normally 0.5-2.0 seconds) whose visual evidence most affected the
decision. Timestamps must be within the clip. Mark whether the distinction requires temporal
context rather than a single still frame.
""".strip()

OCCLUSION_PROMPT = f"""
Analyze this real factory-safety video sampled at 10 frames per second. A short candidate time
window has been replaced by neutral gray frames as a counterfactual temporal occlusion. Classify
the remaining visible evidence exactly once from the taxonomy below. Do not guess what was in
the occluded frames and do not infer a class from filenames or metadata.

{CLASS_GUIDE}

Return probabilities for all eight classes that sum to approximately 1. Critical windows in
this response must refer only to visible evidence that remains in the occluded video.
""".strip()


def response_schema() -> dict[str, Any]:
    labels = list(CANONICAL_LABELS)
    return {
        "type": "object",
        "required": [
            "predicted_label",
            "scores",
            "unsafe",
            "needs_temporal_context",
            "scene_summary",
            "critical_windows",
            "evidence",
        ],
        "properties": {
            "predicted_label": {"type": "string", "enum": labels},
            "scores": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": ["label", "probability"],
                    "properties": {
                        "label": {"type": "string", "enum": labels},
                        "probability": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "unsafe": {"type": "boolean"},
            "needs_temporal_context": {"type": "boolean"},
            "scene_summary": {"type": "string"},
            "critical_windows": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "required": ["start_sec", "end_sec", "reason"],
                    "properties": {
                        "start_sec": {"type": "number", "minimum": 0},
                        "end_sec": {"type": "number", "minimum": 0},
                        "reason": {"type": "string"},
                    },
                },
            },
            "evidence": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string"},
            },
        },
    }


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


class GeminiTeacherStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._budget_lock = threading.Lock()
        with _connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS budget_events (
                    event_id TEXT PRIMARY KEY,
                    clip_id TEXT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_usd REAL NOT NULL DEFAULT 0,
                    estimated_usd REAL NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gemini_runs (
                    clip_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    split TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    proxy_path TEXT NOT NULL,
                    model TEXT NOT NULL,
                    fps REAL NOT NULL,
                    media_resolution TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prediction TEXT,
                    scores_json TEXT,
                    reported_unsafe INTEGER,
                    needs_temporal_context INTEGER,
                    scene TEXT,
                    windows_json TEXT,
                    evidence_json TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_usd REAL NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    raw_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (clip_id, stage)
                );
                CREATE TABLE IF NOT EXISTS everos_pushes (
                    experience_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                """
            )

    def add_prior_spend(self, usd: float, details: dict[str, Any] | None = None) -> None:
        if usd <= 0:
            return
        now = _now()
        with _connect(self.path) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO budget_events
                (event_id, stage, status, estimated_usd, details_json, created_at, updated_at)
                VALUES ('prior-smoke', 'smoke', 'settled', ?, ?, ?, ?)
                """,
                (usd, json.dumps(details or {}, ensure_ascii=False), now, now),
            )

    def budget(self) -> dict[str, float]:
        with _connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN status='settled' THEN estimated_usd ELSE 0 END), 0) spent,
                  COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd ELSE 0 END), 0) reserved
                FROM budget_events
                """
            ).fetchone()
        return {"spent": float(row["spent"]), "reserved": float(row["reserved"])}

    def reserve(self, event_id: str, clip_id: str, stage: str, usd: float, cap: float) -> bool:
        with self._budget_lock, _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT status FROM budget_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing:
                connection.rollback()
                return existing["status"] == "reserved"
            totals = connection.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN status='settled' THEN estimated_usd ELSE 0 END), 0) spent,
                  COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd ELSE 0 END), 0) reserved
                FROM budget_events
                """
            ).fetchone()
            if float(totals["spent"]) + float(totals["reserved"]) + usd > cap + 1e-12:
                connection.rollback()
                return False
            now = _now()
            connection.execute(
                """
                INSERT INTO budget_events
                (event_id, clip_id, stage, status, reserved_usd, created_at, updated_at)
                VALUES (?, ?, ?, 'reserved', ?, ?, ?)
                """,
                (event_id, clip_id, stage, usd, now, now),
            )
            connection.commit()
            return True

    def next_event_id(self, base: str) -> str:
        with _connect(self.path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) count FROM budget_events WHERE event_id LIKE ?",
                (f"{base}:%",),
            ).fetchone()["count"]
        return f"{base}:{int(count) + 1}"

    def settle(
        self,
        event_id: str,
        estimated_usd: float,
        prompt_tokens: int,
        output_tokens: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._budget_lock, _connect(self.path) as connection:
            connection.execute(
                """
                UPDATE budget_events SET status='settled', reserved_usd=0, estimated_usd=?,
                  prompt_tokens=?, output_tokens=?, details_json=?, updated_at=?
                WHERE event_id=?
                """,
                (
                    estimated_usd,
                    prompt_tokens,
                    output_tokens,
                    json.dumps(details or {}, ensure_ascii=False),
                    _now(),
                    event_id,
                ),
            )

    def release(self, event_id: str, error: str) -> None:
        with self._budget_lock, _connect(self.path) as connection:
            connection.execute(
                """
                UPDATE budget_events SET status='failed', reserved_usd=0, details_json=?,
                  updated_at=? WHERE event_id=?
                """,
                (json.dumps({"error": error}, ensure_ascii=False), _now(), event_id),
            )

    def settle_stale_reservations(self) -> int:
        """Conservatively charge abandoned reservations after an interrupted process."""
        with self._budget_lock, _connect(self.path) as connection:
            rows = connection.execute(
                "SELECT event_id,reserved_usd FROM budget_events WHERE status='reserved'"
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE budget_events SET status='settled', estimated_usd=reserved_usd,
                      reserved_usd=0, details_json=?, updated_at=? WHERE event_id=?
                    """,
                    (
                        json.dumps(
                            {
                                "reason": "interrupted_request_conservative_charge",
                                "actual_usage_unknown": True,
                            }
                        ),
                        _now(),
                        row["event_id"],
                    ),
                )
        return len(rows)

    def has_success(self, clip_id: str, stage: str) -> bool:
        with _connect(self.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM gemini_runs WHERE clip_id=? AND stage=? AND status='success'",
                (clip_id, stage),
            ).fetchone()
        return bool(row)

    def success_row(self, clip_id: str, stage: str) -> dict[str, Any] | None:
        with _connect(self.path) as connection:
            row = connection.execute(
                "SELECT * FROM gemini_runs WHERE clip_id=? AND stage=? AND status='success'",
                (clip_id, stage),
            ).fetchone()
        return dict(row) if row else None

    def save_run(self, row: dict[str, Any]) -> None:
        columns = (
            "clip_id",
            "stage",
            "split",
            "label",
            "source_path",
            "proxy_path",
            "model",
            "fps",
            "media_resolution",
            "status",
            "prediction",
            "scores_json",
            "reported_unsafe",
            "needs_temporal_context",
            "scene",
            "windows_json",
            "evidence_json",
            "prompt_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_usd",
            "latency_ms",
            "error",
            "raw_json",
            "updated_at",
        )
        values = [row.get(column) for column in columns]
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns[2:])
        with _connect(self.path) as connection:
            connection.execute(
                f"INSERT INTO gemini_runs ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(clip_id,stage) DO UPDATE SET {updates}",
                values,
            )

    def rows(self, stage: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[str] = []
        if stage:
            clauses.append("stage=?")
            values.append(stage)
        if status:
            clauses.append("status=?")
            values.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with _connect(self.path) as connection:
            rows = connection.execute(
                f"SELECT * FROM gemini_runs{where} ORDER BY clip_id", values
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_everos(self, experience_id: str, status: str, response: dict[str, Any]) -> None:
        with _connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO everos_pushes (experience_id,status,response_json,updated_at)
                VALUES (?,?,?,?) ON CONFLICT(experience_id) DO UPDATE SET
                  status=excluded.status,response_json=excluded.response_json,
                  updated_at=excluded.updated_at
                """,
                (experience_id, status, json.dumps(response, ensure_ascii=False), _now()),
            )

    def everos_done(self, experience_id: str) -> bool:
        with _connect(self.path) as connection:
            row = connection.execute(
                "SELECT status FROM everos_pushes WHERE experience_id=?", (experience_id,)
            ).fetchone()
        return bool(row and row["status"] == "verified")

    def everos_status(self, experience_id: str) -> str | None:
        with _connect(self.path) as connection:
            row = connection.execute(
                "SELECT status FROM everos_pushes WHERE experience_id=?", (experience_id,)
            ).fetchone()
        return str(row["status"]) if row else None

    def summary(self, cap: float) -> dict[str, Any]:
        with _connect(self.path) as connection:
            by_stage = {
                row["stage"]: {"success": int(row["success"]), "failed": int(row["failed"])}
                for row in connection.execute(
                    """
                    SELECT stage,
                      SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) success,
                      SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END) failed
                    FROM gemini_runs GROUP BY stage
                    """
                )
            }
            verified = connection.execute(
                "SELECT COUNT(*) count FROM everos_pushes WHERE status='verified'"
            ).fetchone()["count"]
        budget = self.budget()
        return {
            "budget_cap_usd": cap,
            "estimated_list_spend_usd": round(budget["spent"], 6),
            "active_reservations_usd": round(budget["reserved"], 6),
            "remaining_usd": round(max(0.0, cap - budget["spent"] - budget["reserved"]), 6),
            "runs": by_stage,
            "everos_verified": int(verified),
        }


def _proxy_metadata(source: Path, fps: float, max_width: int) -> dict[str, Any]:
    stat = source.stat()
    return {
        "source": str(source.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "fps": fps,
        "max_width": max_width,
    }


def build_dense_proxy(
    clip: ClipRecord,
    output_dir: Path,
    fps: float = 10.0,
    max_width: int = 960,
) -> Path:
    """Create a real-frame-only, audio-free 10 fps MP4 small enough for inline prompting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{clip.clip_id}.mp4"
    metadata_path = output.with_suffix(".json")
    expected = _proxy_metadata(clip.path, fps, max_width)
    if output.exists() and metadata_path.exists():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            stable_keys = ("source", "source_size", "source_mtime_ns", "fps")
            if (
                all(cached.get(key) == expected[key] for key in stable_keys)
                and int(cached.get("max_width", max_width)) <= max_width
            ):
                return output
        except (OSError, json.JSONDecodeError):
            pass

    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {clip.path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {clip.path}")
    scale = min(1.0, max_width / width)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    temporary = output.with_name(f"{output.stem}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video proxy: {temporary}")
    frame_index = 0
    next_sample_s = 0.0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = frame_index / source_fps
            if timestamp_s + (0.5 / source_fps) >= next_sample_s:
                if (target_width, target_height) != (width, height):
                    frame = cv2.resize(
                        frame, (target_width, target_height), interpolation=cv2.INTER_AREA
                    )
                writer.write(frame)
                written += 1
                next_sample_s += 1.0 / fps
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if written == 0 or not temporary.exists():
        raise RuntimeError(f"No frames written for {clip.path}")
    if temporary.stat().st_size >= 19_000_000:
        temporary.unlink(missing_ok=True)
        if max_width > 640:
            return build_dense_proxy(clip, output_dir, fps=fps, max_width=640)
        raise RuntimeError(f"10 fps proxy is still too large for inline API: {clip.path}")
    temporary.replace(output)
    metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frames / fps if fps > 0 else 0.0


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(0, frames)


def choose_occlusion_window(
    windows: list[dict[str, Any]], duration_s: float
) -> tuple[float, float]:
    if duration_s <= 0:
        return (0.0, 0.0)
    if windows:
        start = max(0.0, min(duration_s, float(windows[0].get("start_sec", 0))))
        end = max(start, min(duration_s, float(windows[0].get("end_sec", start))))
    else:
        start, end = duration_s * 0.375, duration_s * 0.625
    center = (start + end) / 2 if end > start else duration_s / 2
    width = end - start
    max_width = min(2.0, max(0.5, duration_s * 0.35))
    width = min(max_width, max(0.5, width))
    start = max(0.0, min(duration_s - width, center - width / 2))
    return (round(start, 3), round(min(duration_s, start + width), 3))


def build_occluded_proxy(source: Path, output: Path, start_s: float, end_s: float) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not open dense proxy: {source}")
    temporary = output.with_name(f"{output.stem}.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create occluded proxy: {temporary}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = index / fps
            if start_s <= timestamp_s < end_s:
                frame[:] = 127
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()
    if index == 0:
        raise RuntimeError(f"No frames written for occlusion: {source}")
    temporary.replace(output)
    return output


def _normalize_response(text: str, duration_s: float) -> dict[str, Any]:
    value = json.loads(text)
    prediction = str(value["predicted_label"])
    if prediction not in CANONICAL_LABELS:
        raise ValueError(f"Unknown predicted label: {prediction}")
    raw_scores = value.get("scores") or []
    scores = {label: 0.0 for label in CANONICAL_LABELS}
    for row in raw_scores:
        label = str(row.get("label", ""))
        if label in scores:
            scores[label] = max(0.0, float(row.get("probability", 0)))
    total = sum(scores.values())
    if total <= 0:
        scores[prediction] = 1.0
        total = 1.0
    scores = {label: score / total for label, score in scores.items()}
    windows = []
    for row in value.get("critical_windows") or []:
        start = max(0.0, min(duration_s, float(row.get("start_sec", 0))))
        end = max(start, min(duration_s, float(row.get("end_sec", start))))
        windows.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "reason": str(row.get("reason", "")),
            }
        )
    return {
        "predicted_label": prediction,
        "scores": scores,
        "unsafe": bool(value.get("unsafe", prediction in UNSAFE_LABELS)),
        "needs_temporal_context": bool(value.get("needs_temporal_context", False)),
        "scene_summary": str(value.get("scene_summary", "")),
        "critical_windows": windows[:3],
        "evidence": [str(item) for item in (value.get("evidence") or [])][:5],
    }


class GeminiAnalyzer:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._local = threading.local()

    def client(self) -> genai.Client:
        client = getattr(self._local, "client", None)
        if client is None:
            client = genai.Client(api_key=self.api_key)
            self._local.client = client
        return client

    @staticmethod
    def contents(proxy: Path, prompt: str, fps: float) -> list[Any]:
        part = types.Part(
            inline_data=types.Blob(data=proxy.read_bytes(), mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=fps),
        )
        return [part, prompt]

    def count_tokens(self, model: str, contents: list[Any]) -> int:
        for attempt in range(3):
            try:
                response = self.client().models.count_tokens(model=model, contents=contents)
                return int(response.total_tokens or 0)
            except Exception as exc:
                if attempt == 2 or not any(code in str(exc) for code in ("429", "500", "503")):
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Unreachable count_tokens retry state")

    def generate(
        self,
        model: str,
        contents: list[Any],
        media_resolution: str,
        thinking_level: str,
    ) -> tuple[dict[str, Any], dict[str, int], float, str]:
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=response_schema(),
            media_resolution=media_resolution,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )
        started = time.perf_counter()
        response = None
        for attempt in range(3):
            try:
                response = self.client().models.generate_content(
                    model=model, contents=contents, config=config
                )
                break
            except Exception as exc:
                if attempt == 2 or not any(code in str(exc) for code in ("429", "500", "503")):
                    raise
                time.sleep(2**attempt)
        if response is None:
            raise RuntimeError("Unreachable generate retry state")
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage_metadata
        tokens = {
            "prompt": int(getattr(usage, "prompt_token_count", 0) or 0),
            "candidates": int(getattr(usage, "candidates_token_count", 0) or 0),
            "thoughts": int(getattr(usage, "thoughts_token_count", 0) or 0),
            "total": int(getattr(usage, "total_token_count", 0) or 0),
        }
        return {}, tokens, latency_ms, response.text


def _run_row(
    clip: ClipRecord,
    proxy: Path,
    stage: str,
    model: str,
    fps: float,
    resolution: str,
    status: str,
    parsed: dict[str, Any] | None = None,
    tokens: dict[str, int] | None = None,
    estimated_usd: float = 0.0,
    latency_ms: float = 0.0,
    error: str | None = None,
    raw: str | None = None,
) -> dict[str, Any]:
    parsed = parsed or {}
    tokens = tokens or {}
    return {
        "clip_id": clip.clip_id,
        "stage": stage,
        "split": clip.split,
        "label": clip.label,
        "source_path": str(clip.path),
        "proxy_path": str(proxy),
        "model": model,
        "fps": fps,
        "media_resolution": resolution,
        "status": status,
        "prediction": parsed.get("predicted_label"),
        "scores_json": json.dumps(parsed.get("scores", {}), ensure_ascii=False),
        "reported_unsafe": int(bool(parsed.get("unsafe"))) if parsed else None,
        "needs_temporal_context": (
            int(bool(parsed.get("needs_temporal_context"))) if parsed else None
        ),
        "scene": parsed.get("scene_summary"),
        "windows_json": json.dumps(parsed.get("critical_windows", []), ensure_ascii=False),
        "evidence_json": json.dumps(parsed.get("evidence", []), ensure_ascii=False),
        "prompt_tokens": int(tokens.get("prompt", 0)),
        "output_tokens": int(tokens.get("candidates", 0)) + int(tokens.get("thoughts", 0)),
        "total_tokens": int(tokens.get("total", 0)),
        "estimated_usd": estimated_usd,
        "latency_ms": latency_ms,
        "error": error,
        "raw_json": raw,
        "updated_at": _now(),
    }


def _analyze_reserved(
    analyzer: GeminiAnalyzer,
    store: GeminiTeacherStore,
    clip: ClipRecord,
    proxy: Path,
    stage: str,
    model: str,
    prompt: str,
    fps: float,
    resolution: str,
    thinking: str,
    budget_cap: float,
) -> str:
    if store.has_success(clip.clip_id, stage):
        return "skipped"
    contents = analyzer.contents(proxy, prompt, fps)
    try:
        input_tokens = analyzer.count_tokens(model, contents)
    except Exception as exc:
        store.save_run(
            _run_row(
                clip, proxy, stage, model, fps, resolution, "failed", error=f"count_tokens: {exc}"
            )
        )
        return "failed"
    price = MODEL_PRICES[model]
    reserve_usd = price.estimate(input_tokens, MAX_OUTPUT_TOKENS)
    event_id = store.next_event_id(f"{stage}:{clip.clip_id}")
    if not store.reserve(event_id, clip.clip_id, stage, reserve_usd, budget_cap):
        return "budget_stop"
    billed = False
    try:
        _, tokens, latency_ms, raw = analyzer.generate(model, contents, resolution, thinking)
        output_tokens = tokens["candidates"] + tokens["thoughts"]
        cost = price.estimate(tokens["prompt"], output_tokens)
        store.settle(event_id, cost, tokens["prompt"], output_tokens)
        billed = True
        parsed = _normalize_response(raw, video_duration(proxy))
        store.save_run(
            _run_row(
                clip,
                proxy,
                stage,
                model,
                fps,
                resolution,
                "success",
                parsed,
                tokens,
                cost,
                latency_ms,
                raw=raw,
            )
        )
        return "success"
    except Exception as exc:
        if not billed:
            store.release(event_id, f"{type(exc).__name__}: {exc}")
        store.save_run(
            _run_row(
                clip,
                proxy,
                stage,
                model,
                fps,
                resolution,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return "failed"


def build_proxies(
    clips: Iterable[ClipRecord],
    output_dir: Path,
    fps: float,
    workers: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    rows = list(clips)
    proxies: dict[str, Path] = {}
    failures: list[dict[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(build_dense_proxy, clip, output_dir, fps): clip for clip in rows}
        for future in as_completed(futures):
            clip = futures[future]
            try:
                proxies[clip.clip_id] = future.result()
                status = "proxy_ready"
            except Exception as exc:
                failures.append({"clip_id": clip.clip_id, "error": str(exc)})
                status = "proxy_failed"
            done += 1
            if progress:
                progress(done, len(rows), status)
    return proxies, failures


def run_cheap_pass(
    clips: Iterable[ClipRecord],
    proxies: dict[str, Path],
    analyzer: GeminiAnalyzer,
    store: GeminiTeacherStore,
    fps: float,
    budget_cap: float,
    workers: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    rows = [clip for clip in clips if clip.clip_id in proxies]
    counts: defaultdict[str, int] = defaultdict(int)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _analyze_reserved,
                analyzer,
                store,
                clip,
                proxies[clip.clip_id],
                "cheap_dense",
                CHEAP_MODEL,
                BASE_PROMPT,
                fps,
                "MEDIA_RESOLUTION_LOW",
                "minimal",
                budget_cap,
            ): clip
            for clip in rows
        }
        for future in as_completed(futures):
            status = future.result()
            counts[status] += 1
            done += 1
            if progress:
                progress(done, len(rows), f"cheap_{status}")
    return dict(counts)


def _priority(row: dict[str, Any]) -> tuple[float, ...]:
    scores = json.loads(row.get("scores_json") or "{}")
    prediction = row.get("prediction") or ""
    label = row["label"]
    predicted_unsafe = prediction in UNSAFE_LABELS
    false_negative = label in UNSAFE_LABELS and not predicted_unsafe
    incorrect = prediction != label
    confidence = float(scores.get(prediction, 0))
    entropy = -sum(float(value) * math.log(max(float(value), 1e-9)) for value in scores.values())
    return (
        float(false_negative),
        float(incorrect),
        float(bool(row.get("needs_temporal_context"))),
        1.0 - confidence,
        entropy,
    )


def prioritized_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    false_negatives = [row for row in rows if _priority(row)[0] > 0]
    false_negatives.sort(key=_priority, reverse=True)
    selected_ids = {row["clip_id"] for row in false_negatives}
    buckets: dict[str, deque[dict[str, Any]]] = {}
    for label in CANONICAL_LABELS:
        candidates = [
            row for row in rows if row["label"] == label and row["clip_id"] not in selected_ids
        ]
        candidates.sort(key=_priority, reverse=True)
        buckets[label] = deque(candidates)
    balanced: list[dict[str, Any]] = []
    label_order = sorted(CANONICAL_LABELS, key=lambda label: label not in UNSAFE_LABELS)
    while any(buckets.values()):
        for label in label_order:
            if buckets[label]:
                balanced.append(buckets[label].popleft())
    return false_negatives + balanced


def _row_to_clip(row: dict[str, Any]) -> ClipRecord:
    return ClipRecord(
        clip_id=row["clip_id"],
        path=Path(row["source_path"]),
        label=row["label"],
        split=row["split"],
        size_bytes=Path(row["source_path"]).stat().st_size,
    )


def run_strong_counterfactuals(
    analyzer: GeminiAnalyzer,
    store: GeminiTeacherStore,
    work_dir: Path,
    fps: float,
    budget_cap: float,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    cheap_rows = prioritized_candidates(store.rows(stage="cheap_dense", status="success"))
    counts: defaultdict[str, int] = defaultdict(int)
    attempted = 0
    for cheap in cheap_rows:
        clip = _row_to_clip(cheap)
        if store.has_success(clip.clip_id, "pro_full") and store.has_success(
            clip.clip_id, "pro_occluded"
        ):
            counts["skipped"] += 1
            continue
        proxy = Path(cheap["proxy_path"])
        price = MODEL_PRICES[STRONG_MODEL]
        existing_full_row = store.success_row(clip.clip_id, "pro_full")
        frame_upper_tokens = video_frame_count(proxy) * 300 + 3_000
        local_floor = price.estimate(frame_upper_tokens, MAX_OUTPUT_TOKENS)
        required_floor = local_floor if existing_full_row else 2 * local_floor
        budget = store.budget()
        if budget["spent"] + budget["reserved"] + required_floor > budget_cap + 1e-12:
            counts["budget_skipped"] += 1
            continue
        full_contents = analyzer.contents(proxy, BASE_PROMPT, fps)
        try:
            count = analyzer.count_tokens(STRONG_MODEL, full_contents)
        except Exception as exc:
            counts["failed"] += 1
            store.save_run(
                _run_row(
                    clip,
                    proxy,
                    "pro_full",
                    STRONG_MODEL,
                    fps,
                    "MEDIA_RESOLUTION_HIGH",
                    "failed",
                    error=f"count_tokens: {exc}",
                )
            )
            continue
        conservative_input = max(count, video_frame_count(proxy) * 300 + 3_000)
        request_reserve = price.estimate(conservative_input, MAX_OUTPUT_TOKENS)
        # Reserve a complete teacher + temporal-occlusion pair before the first billable call.
        pair_reserve = request_reserve if existing_full_row else 2 * request_reserve
        event_id = store.next_event_id(f"pro_pair:{clip.clip_id}")
        if not store.reserve(event_id, clip.clip_id, "pro_pair", pair_reserve, budget_cap):
            # A later, shorter clip can still fit even when this candidate cannot.
            counts["budget_skipped"] += 1
            continue
        attempted += 1
        pair_cost = 0.0
        pair_prompt_tokens = 0
        pair_output_tokens = 0
        try:
            duration = video_duration(proxy)
            if existing_full_row:
                full = _decoded(existing_full_row)
            else:
                _, full_tokens, full_latency, full_raw = analyzer.generate(
                    STRONG_MODEL, full_contents, "MEDIA_RESOLUTION_HIGH", "low"
                )
                full_output = full_tokens["candidates"] + full_tokens["thoughts"]
                full_cost = price.estimate(full_tokens["prompt"], full_output)
                pair_cost += full_cost
                pair_prompt_tokens += full_tokens["prompt"]
                pair_output_tokens += full_output
                full = _normalize_response(full_raw, duration)
                store.save_run(
                    _run_row(
                        clip,
                        proxy,
                        "pro_full",
                        STRONG_MODEL,
                        fps,
                        "MEDIA_RESOLUTION_HIGH",
                        "success",
                        full,
                        full_tokens,
                        full_cost,
                        full_latency,
                        raw=full_raw,
                    )
                )

            start_s, end_s = choose_occlusion_window(full["critical_windows"], duration)
            occluded = work_dir / "occluded" / f"{clip.clip_id}.mp4"
            build_occluded_proxy(proxy, occluded, start_s, end_s)
            occlusion_prompt = (
                f"{OCCLUSION_PROMPT}\n\nThe intentionally occluded interval is "
                f"{start_s:.3f}-{end_s:.3f} seconds."
            )
            occluded_contents = analyzer.contents(occluded, occlusion_prompt, fps)
            occluded_count = analyzer.count_tokens(STRONG_MODEL, occluded_contents)
            occluded_conservative_input = max(
                occluded_count, video_frame_count(occluded) * 300 + 3_000
            )
            occluded_worst = price.estimate(occluded_conservative_input, MAX_OUTPUT_TOKENS)
            remaining_reservation = request_reserve
            if occluded_worst > remaining_reservation + 1e-12:
                raise RuntimeError("Counterfactual request exceeded conservative pair reservation")
            _, occluded_tokens, occluded_latency, occluded_raw = analyzer.generate(
                STRONG_MODEL, occluded_contents, "MEDIA_RESOLUTION_HIGH", "low"
            )
            occluded_output = occluded_tokens["candidates"] + occluded_tokens["thoughts"]
            occluded_cost = price.estimate(occluded_tokens["prompt"], occluded_output)
            pair_cost += occluded_cost
            pair_prompt_tokens += occluded_tokens["prompt"]
            pair_output_tokens += occluded_output
            occluded_result = _normalize_response(occluded_raw, duration)
            occluded_result["occluded_window"] = {"start_sec": start_s, "end_sec": end_s}
            store.save_run(
                _run_row(
                    clip,
                    occluded,
                    "pro_occluded",
                    STRONG_MODEL,
                    fps,
                    "MEDIA_RESOLUTION_HIGH",
                    "success",
                    occluded_result,
                    occluded_tokens,
                    occluded_cost,
                    occluded_latency,
                    raw=occluded_raw,
                )
            )
            store.settle(
                event_id,
                pair_cost,
                pair_prompt_tokens,
                pair_output_tokens,
                {"occluded_window": [start_s, end_s]},
            )
            counts["success"] += 1
            status = "pro_pair_success"
        except Exception as exc:
            if pair_cost > 0:
                store.settle(
                    event_id,
                    pair_cost,
                    pair_prompt_tokens,
                    pair_output_tokens,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
            else:
                store.release(event_id, f"{type(exc).__name__}: {exc}")
            counts["failed"] += 1
            status = "pro_pair_failed"
        if progress:
            progress(attempted, len(cheap_rows), status)
    return dict(counts)


def _decoded(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["scores"] = json.loads(row.get("scores_json") or "{}")
    output["critical_windows"] = json.loads(row.get("windows_json") or "[]")
    output["evidence"] = json.loads(row.get("evidence_json") or "[]")
    return output


def build_experiences(store: GeminiTeacherStore) -> list[dict[str, Any]]:
    cheap = {row["clip_id"]: _decoded(row) for row in store.rows("cheap_dense", "success")}
    full = {row["clip_id"]: _decoded(row) for row in store.rows("pro_full", "success")}
    occluded = {row["clip_id"]: _decoded(row) for row in store.rows("pro_occluded", "success")}
    experiences: list[dict[str, Any]] = []
    for clip_id in sorted(cheap):
        baseline = cheap[clip_id]
        label = baseline["label"]
        baseline_prediction = baseline.get("prediction") or ""
        baseline_correct = baseline_prediction == label
        unsafe_false_negative = label in UNSAFE_LABELS and baseline_prediction not in UNSAFE_LABELS
        baseline_gt = max(float(baseline["scores"].get(label, 0)), 1e-9)
        observations = [
            {
                "action": "flash_lite_dense_10fps",
                "prediction": baseline_prediction,
                "correct": baseline_correct,
                "gt_probability": baseline_gt,
                "unsafe_false_negative": unsafe_false_negative,
                "estimated_usd": baseline.get("estimated_usd", 0),
            }
        ]
        metadata: dict[str, Any] = {
            "provider": "google_gemini_developer_api",
            "prompt_version": PROMPT_VERSION,
            "experience_version": EXPERIENCE_VERSION,
            "fps": 10,
            "ground_truth_source": "dataset_directory_label",
            "unsafe_false_negative": unsafe_false_negative,
            "counterfactual_tested": False,
            "window_validated": None,
        }

        if clip_id in full and clip_id in occluded:
            teacher = full[clip_id]
            counterfactual = occluded[clip_id]
            full_gt = max(float(teacher["scores"].get(label, 0)), 1e-9)
            occluded_gt = max(float(counterfactual["scores"].get(label, 0)), 1e-9)
            loss_increase = math.log(full_gt / occluded_gt)
            flip = teacher["prediction"] != counterfactual["prediction"]
            unsafe_failure = (
                label in UNSAFE_LABELS
                and teacher["prediction"] in UNSAFE_LABELS
                and counterfactual["prediction"] not in UNSAFE_LABELS
            )
            validated = loss_increase > 0.15 or flip or unsafe_failure
            duration = video_duration(Path(teacher["proxy_path"]))
            start_s, end_s = choose_occlusion_window(teacher["critical_windows"], duration)
            window = {"start_sec": start_s, "end_sec": end_s}
            observations.extend(
                [
                    {
                        "action": "pro_dense_10fps",
                        "prediction": teacher["prediction"],
                        "correct": teacher["prediction"] == label,
                        "gt_probability": full_gt,
                        "estimated_usd": teacher["estimated_usd"],
                    },
                    {
                        "action": "pro_temporal_occlusion",
                        "occluded_window": window,
                        "prediction": counterfactual["prediction"],
                        "correct": counterfactual["prediction"] == label,
                        "gt_probability": occluded_gt,
                        "estimated_usd": counterfactual["estimated_usd"],
                    },
                ]
            )
            if teacher["prediction"] == label:
                best_action = (
                    "pro_dense_10fps_use_validated_window"
                    if validated
                    else "pro_dense_10fps_ignore_rejected_window"
                )
            elif baseline_correct:
                best_action = "flash_lite_dense_10fps_with_gt_guardrail"
            else:
                best_action = "abstain_and_escalate_with_gt_guardrail"
            window_directive = (
                "retain this interval as causal temporal evidence"
                if validated
                else "reject this interval as a causal shortcut"
            )
            lesson = (
                f"GT={label}; occluding {start_s:.3f}-{end_s:.3f}s changed GT probability "
                f"from {full_gt:.4f} to {occluded_gt:.4f} (log-loss increase "
                f"{loss_increase:.4f}, prediction flip={flip}, unsafe failure={unsafe_failure}). "
                f"Therefore {window_directive}; selected policy={best_action}."
            )
            outcome = "counterfactual_strategy_audit_completed"
            scene = teacher.get("scene") or baseline.get("scene") or ""
            metadata.update(
                {
                    "teacher_model": STRONG_MODEL,
                    "counterfactual_tested": True,
                    "critical_window": window,
                    "window_validated": validated,
                    "gt_log_loss_increase": loss_increase,
                    "prediction_flip": flip,
                    "unsafe_failure_induced": unsafe_failure,
                }
            )
        else:
            if unsafe_false_negative:
                best_action = "abstain_and_escalate_unsafe_false_negative"
                outcome = "unsafe_escalation_rule_verified"
                directive = "never accept the cheap safe prediction; escalate or abstain"
            elif baseline_correct:
                best_action = "flash_lite_dense_10fps_with_gt_guardrail"
                outcome = "cheap_strategy_verified"
                directive = "the cheap path is eligible only under the same validated guardrails"
            else:
                best_action = "abstain_and_escalate_classification_error"
                outcome = "classification_escalation_rule_verified"
                directive = "reject the cheap prediction and escalate or abstain"
            lesson = (
                f"GT={label}; Flash-Lite predicted {baseline_prediction} at 10 fps with "
                f"GT probability {baseline_gt:.4f}. Ground-truth evaluation says to {directive}."
            )
            scene = baseline.get("scene") or ""

        text = (
            f"Factory-safety strategy audit completed for GT={label}. {lesson} "
            "The stored outcome is an evaluated routing rule, not an unverified model answer."
        )
        experience_id = hashlib.sha256(
            f"{EXPERIENCE_VERSION}:{clip_id}:gt-evaluated".encode()
        ).hexdigest()[:24]
        experiences.append(
            {
                "experience_id": experience_id,
                "payload": {
                    "clip_id": clip_id,
                    "label": label,
                    "split": baseline["split"],
                    "scene": scene,
                    "observations": observations,
                    "best_action": best_action,
                    "key_insight": lesson,
                    "outcome": outcome,
                    "task_completed": True,
                    "quality_score": 1.0,
                    "text": text,
                    "metadata": metadata,
                },
            }
        )
    return experiences


def export_experiences(store: GeminiTeacherStore, output: Path) -> int:
    experiences = build_experiences(store)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in experiences:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(experiences)


def load_experiences(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
