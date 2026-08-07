from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from typing import Any

from everos_cloud import EverOS

from .config import require_env


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        data = value.model_dump(mode="json")
        return data if isinstance(data, dict) else {"data": data}
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        return data if isinstance(data, dict) else {"data": data}
    try:
        data = json.loads(str(value))
        return data if isinstance(data, dict) else {"data": data}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"data": str(value)}


class EverOSClient:
    """Thread-safe façade over the official everos-cloud SDK."""

    def __init__(self, timeout_s: float = 60.0) -> None:
        required = require_env("EVEROS_API_KEY")
        self.api_key = required["EVEROS_API_KEY"]
        self.host = (
            os.getenv("EVER_OS_BASE_URL")
            or os.getenv("EVEROS_BASE_URL")
            or "https://api.evermind.ai"
        ).rstrip("/")
        self.user_id = os.getenv("EVEROS_USER_ID", "mavis-hackathon")
        self.agent_id = os.getenv("EVEROS_AGENT_ID", "mavis-agent-v1")
        self.app_id = os.getenv("EVEROS_APP_ID", "mavis")
        self.project_id = os.getenv("EVEROS_PROJECT_ID", "factory-safety-seed-v1")
        self.timeout_s = timeout_s
        self._local = threading.local()
        self._clients: list[EverOS] = []
        self._clients_lock = threading.Lock()

    def _sdk(self) -> EverOS:
        client = getattr(self._local, "client", None)
        if client is None:
            client = EverOS(
                api_key=self.api_key,
                host=self.host,
                app_id=self.app_id,
                project_id=self.project_id,
                timeout=self.timeout_s,
            )
            self._local.client = client
            with self._clients_lock:
                self._clients.append(client)
        return client

    def add_experience(self, row: dict[str, Any], session_id: str) -> dict[str, Any]:
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        messages = self._experience_messages(row, now_ms)
        return _plain(
            self._sdk().add(
                session_id=session_id,
                messages=messages,
                mode="agent",
                async_mode=False,
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def add_durable_experience(self, row: dict[str, Any], session_id: str) -> dict[str, Any]:
        """Store one evaluated experience on the reliably retrievable user-memory track."""
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        payload = row["payload"]
        metadata = payload.get("metadata") or {}
        counterfactual = (
            f" counterfactual_tested={metadata.get('counterfactual_tested')};"
            f" critical_window={metadata.get('critical_window')};"
            f" window_validated={metadata.get('window_validated')};"
            f" gt_log_loss_increase={metadata.get('gt_log_loss_increase')};"
            if metadata.get("counterfactual_tested")
            else ""
        )
        content = (
            f"Authoritative evaluated factory-safety experience {row['experience_id']}. "
            f"clip={payload['clip_id']}; ground_truth={payload['label']}; "
            f"outcome={payload['outcome']}; best_action={payload['best_action']};"
            f"{counterfactual} key_insight={payload['key_insight']}"
        )
        messages = [
            {
                "sender_id": self.user_id,
                "role": "user",
                "timestamp": now_ms,
                "content": content,
            },
            {
                "sender_id": self.agent_id,
                "role": "assistant",
                "timestamp": now_ms + 1,
                "content": (
                    "Recorded this ground-truth-evaluated routing experience for future "
                    "retrieval. It is a measured policy lesson, not an unverified prediction."
                ),
            },
        ]
        return _plain(
            self._sdk().add(
                session_id=session_id,
                messages=messages,
                mode="chat",
                async_mode=False,
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def _experience_messages(self, row: dict[str, Any], now_ms: int) -> list[dict[str, Any]]:
        payload = row["payload"]
        observations = json.dumps(
            payload["observations"], ensure_ascii=False, separators=(",", ":")
        )
        tool_calls = [
            {
                "id": f"visual_trial_{index}",
                "type": "function",
                "function": {
                    "name": "run_measured_visual_trial",
                    "arguments": json.dumps(
                        {
                            "clip_id": payload["clip_id"],
                            "action": observation.get("action"),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for index, observation in enumerate(payload["observations"], start=1)
        ]
        evaluation_call_id = "evaluate_ground_truth"
        tool_calls.append(
            {
                "id": evaluation_call_id,
                "type": "function",
                "function": {
                    "name": "evaluate_strategy_against_dataset_ground_truth",
                    "arguments": json.dumps(
                        {
                            "clip_id": payload["clip_id"],
                            "ground_truth": payload["label"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
        messages = [
            {
                "sender_id": self.user_id,
                "role": "user",
                "timestamp": now_ms,
                "content": (
                    "Task: determine the lowest-cost correct visual strategy for factory safety. "
                    f"Clip: {payload['clip_id']}. Primary clip label: {payload['label']}."
                ),
            },
            {
                "sender_id": self.agent_id,
                "role": "assistant",
                "timestamp": now_ms + 1,
                "content": "",
                "tool_calls": tool_calls,
            },
            *[
                {
                    "sender_id": "mavis-visual-evaluator",
                    "role": "tool",
                    "timestamp": now_ms + index + 1,
                    "tool_call_id": f"visual_trial_{index}",
                    "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
                }
                for index, observation in enumerate(payload["observations"], start=1)
            ],
            {
                "sender_id": "mavis-ground-truth-evaluator",
                "role": "tool",
                "timestamp": now_ms + len(payload["observations"]) + 2,
                "tool_call_id": evaluation_call_id,
                "content": json.dumps(
                    {
                        "ground_truth": payload["label"],
                        "outcome": payload.get("outcome"),
                        "best_action": payload.get("best_action"),
                        "task_completed": payload.get("task_completed", True),
                        "quality_score": payload.get("quality_score", 1.0),
                        "key_insight": payload.get("key_insight"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {
                "sender_id": self.agent_id,
                "role": "assistant",
                "timestamp": now_ms + len(payload["observations"]) + 3,
                "content": (
                    "Task completed successfully. "
                    f"Measured observations: {observations}\n"
                    f"Verified conclusion: {payload['text']}"
                ),
            },
        ]
        return messages

    def add_experience_batch(self, rows: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
        """Let EverOS boundary detection split a realistic multi-task agent session."""
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        messages: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            messages.extend(self._experience_messages(row, now_ms + index * 20))
        if len(messages) > 500:
            raise ValueError("EverOS accepts at most 500 messages per add call")
        return _plain(
            self._sdk().add(
                session_id=session_id,
                messages=messages,
                mode="agent",
                async_mode=False,
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    @staticmethod
    def _outcome_feedback(payload: dict[str, Any]) -> str:
        metadata = payload.get("metadata") or {}
        return (
            "Final evaluator feedback. "
            f"Dataset ground truth: {payload['label']}. "
            f"Measured trajectory outcome: {payload.get('outcome', 'unknown')}. "
            f"Critical-window validation: {metadata.get('window_validated', 'unknown')}. "
            f"Unsafe failure induced by occlusion: "
            f"{metadata.get('unsafe_failure_induced', 'unknown')}. "
            f"Verified lesson: {payload.get('key_insight', payload['text'])}"
        )

    def add_outcome_feedback(self, row: dict[str, Any], session_id: str) -> dict[str, Any]:
        """Complete an already-flushed agent trajectory without replaying prior messages."""
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        payload = row["payload"]
        return _plain(
            self._sdk().add(
                session_id=session_id,
                messages=[
                    {
                        "sender_id": self.user_id,
                        "role": "user",
                        "timestamp": now_ms,
                        "content": self._outcome_feedback(payload),
                    }
                ],
                mode="agent",
                async_mode=False,
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def flush(self, session_id: str) -> dict[str, Any]:
        return _plain(
            self._sdk().flush(
                session_id,
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        scope: str = "user",
    ) -> dict[str, Any]:
        if scope not in {"user", "agent"}:
            raise ValueError("EverOS search scope must be user or agent")
        owner = {"user_id": self.user_id} if scope == "user" else {"agent_id": self.agent_id}
        return _plain(
            self._sdk().search(
                query,
                **owner,
                method=method,
                top_k=top_k,
                enable_llm_rerank=method == "hybrid",
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def get(self, memory_type: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        if memory_type not in {"episode", "profile", "agent_case", "agent_skill"}:
            raise ValueError("MAVIS memory supports episode, profile, agent_case, or agent_skill")
        owner = (
            {"user_id": self.user_id}
            if memory_type in {"episode", "profile"}
            else {"agent_id": self.agent_id}
        )
        return _plain(
            self._sdk().get(
                memory_type,
                **owner,
                page=page,
                page_size=page_size,
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def get_session_episodes(self, session_id: str) -> dict[str, Any]:
        return _plain(
            self._sdk().get(
                "episode",
                user_id=self.user_id,
                page=1,
                page_size=10,
                filters={"session_id": session_id},
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def get_session_cases(self, session_id: str) -> dict[str, Any]:
        return _plain(
            self._sdk().get(
                "agent_case",
                agent_id=self.agent_id,
                page=1,
                page_size=10,
                filters={"session_id": session_id},
                app_id=self.app_id,
                project_id=self.project_id,
            )
        )

    def close(self) -> None:
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.close()

    def __enter__(self) -> EverOSClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
