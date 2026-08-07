from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .evaluation import build_experience, evaluate_actions
from .models import Experience, InferenceResult

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS experiences (
    experience_id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    split TEXT NOT NULL,
    label TEXT NOT NULL,
    outcome TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    everos_status TEXT,
    everos_response_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_experience_clip_version
ON experiences(clip_id, prompt_version);

CREATE TABLE IF NOT EXISTS action_runs (
    experience_id TEXT NOT NULL,
    clip_id TEXT NOT NULL,
    action TEXT NOT NULL,
    model TEXT NOT NULL,
    query_id TEXT,
    prediction TEXT,
    correct INTEGER NOT NULL,
    information_gain REAL NOT NULL,
    latency_ms REAL NOT NULL,
    estimated_credits REAL,
    actual_credits REAL,
    error TEXT,
    raw_response_json TEXT NOT NULL,
    PRIMARY KEY (experience_id, action),
    FOREIGN KEY (experience_id) REFERENCES experiences(experience_id)
);

CREATE INDEX IF NOT EXISTS ix_action_query ON action_runs(query_id);
"""


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def is_complete(self, clip_id: str, prompt_version: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM experiences WHERE clip_id=? AND prompt_version=?",
                (clip_id, prompt_version),
            ).fetchone()
            return row is not None

    def save_experience(self, experience: Experience) -> None:
        prompt_version = str(experience.metadata["prompt_version"])
        payload = json.dumps(experience.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR REPLACE INTO experiences
                (experience_id, clip_id, prompt_version, split, label, outcome, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    experience.experience_id,
                    experience.clip_id,
                    prompt_version,
                    experience.split,
                    experience.label,
                    experience.outcome,
                    payload,
                ),
            )
            for item in experience.observations:
                result = item.result
                connection.execute(
                    """INSERT OR REPLACE INTO action_runs
                    (experience_id, clip_id, action, model, query_id, prediction, correct,
                     information_gain, latency_ms, estimated_credits, actual_credits, error,
                     raw_response_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        experience.experience_id,
                        experience.clip_id,
                        result.action,
                        result.model,
                        result.query_id,
                        result.prediction,
                        int(item.correct),
                        item.information_gain,
                        result.latency_ms,
                        result.estimated_credits,
                        result.actual_credits,
                        result.error,
                        json.dumps(result.raw_response, ensure_ascii=False),
                    ),
                )
            connection.commit()

    def mark_everos(self, experience_id: str, status: str, response: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE experiences SET everos_status=?, everos_response_json=? WHERE experience_id=?",
                (status, json.dumps(response, ensure_ascii=False), experience_id),
            )
            connection.commit()

    def pending_everos(self, limit: int | None = None) -> list[dict]:
        sql = """SELECT e.experience_id, e.payload_json
        FROM experiences e
        WHERE e.everos_status IS NULL
          AND e.split = 'seed'
          AND json_extract(e.payload_json, '$.metadata.evidence_source') = 'snowflake'
          AND NOT EXISTS (
            SELECT 1 FROM action_runs a
            WHERE a.experience_id = e.experience_id
              AND (a.actual_credits IS NULL OR a.error IS NOT NULL)
          )
        ORDER BY e.created_at"""
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as connection:
            return [
                {"experience_id": row["experience_id"], "payload": json.loads(row["payload_json"])}
                for row in connection.execute(sql, params)
            ]

    def query_ids_without_actual_cost(self) -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT query_id FROM action_runs "
                    "WHERE query_id IS NOT NULL AND actual_credits IS NULL"
                )
            ]

    def experiences_by_clip_ids(self, clip_ids: list[str]) -> dict[str, dict]:
        ids = sorted(set(clip_ids))
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT clip_id, payload_json FROM experiences WHERE clip_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row["clip_id"]: json.loads(row["payload_json"]) for row in rows}

    def update_actual_costs(self, costs: dict[str, float]) -> int:
        updated = 0
        with self._lock, self._connect() as connection:
            for query_id, credits in costs.items():
                cursor = connection.execute(
                    "UPDATE action_runs SET actual_credits=? WHERE query_id=?",
                    (credits, query_id),
                )
                updated += cursor.rowcount
            connection.commit()
        return updated

    def rebuild_reconciled(self, labels: tuple[str, ...]) -> int:
        """Re-evaluate provisional trajectories after billed query credits arrive."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT e.payload_json
                FROM experiences e
                WHERE NOT EXISTS (
                  SELECT 1 FROM action_runs a
                  WHERE a.experience_id=e.experience_id
                    AND (a.actual_credits IS NULL OR a.error IS NOT NULL)
                )"""
            ).fetchall()
        rebuilt = 0
        for row in rows:
            payload = json.loads(row["payload_json"])
            with self._connect() as connection:
                costs = {
                    action["action"]: action["actual_credits"]
                    for action in connection.execute(
                        "SELECT action, actual_credits FROM action_runs WHERE experience_id=?",
                        (payload["experience_id"],),
                    )
                }
            results = []
            for observation in payload["observations"]:
                result_fields = {
                    key: observation.get(key) for key in InferenceResult.__dataclass_fields__
                }
                result_fields["actual_credits"] = costs[observation["action"]]
                results.append(InferenceResult(**result_fields))
            evaluated = evaluate_actions(payload["label"], results, labels)
            experience = build_experience(
                payload["clip_id"],
                payload["label"],
                payload["split"],
                evaluated,
                payload["metadata"]["prompt_version"],
            )
            experience = Experience(
                **{
                    **experience.__dict__,
                    "metadata": {
                        **experience.metadata,
                        **payload["metadata"],
                        "actual_costs_complete": True,
                    },
                }
            )
            self.save_experience(experience)
            rebuilt += 1
        return rebuilt

    def summary(self) -> dict[str, object]:
        with self._connect() as connection:
            experience = connection.execute(
                "SELECT COUNT(*) n, SUM(outcome='success') successes FROM experiences"
            ).fetchone()
            action = connection.execute(
                """SELECT COUNT(*) n,
                SUM(CASE WHEN actual_credits IS NOT NULL THEN actual_credits ELSE 0 END) credits,
                SUM(actual_credits IS NOT NULL) reconciled,
                SUM(error IS NOT NULL) errors
                FROM action_runs"""
            ).fetchone()
            by_best = connection.execute(
                "SELECT json_extract(payload_json, '$.best_action') action, COUNT(*) n "
                "FROM experiences GROUP BY action"
            ).fetchall()
            return {
                "experiences": experience["n"] or 0,
                "successful_experiences": experience["successes"] or 0,
                "actions": action["n"] or 0,
                "errors": action["errors"] or 0,
                "actual_credits_reconciled_actions": action["reconciled"] or 0,
                "actual_credits": action["credits"] or 0.0,
                "best_actions": {row["action"]: row["n"] for row in by_best},
            }
