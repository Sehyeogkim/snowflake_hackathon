from __future__ import annotations

from typing import Any, ClassVar

import mavis.everos as everos_module
from mavis.everos import EverOSClient


class FakeSDK:
    instances: ClassVar[list[FakeSDK]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.instances.append(self)

    def add(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("add", kwargs))
        return {"status": "queued"}

    def flush(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("flush", {"session_id": session_id, **kwargs}))
        return {"status": "extracted"}

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", {"query": query, **kwargs}))
        return {"data": {"agent_cases": [], "agent_skills": []}}

    def get(self, memory_type: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", {"memory_type": memory_type, **kwargs}))
        return {"data": {"agent_cases": [], "agent_skills": []}}

    def close(self) -> None:
        pass


def test_agent_memory_contract(monkeypatch) -> None:
    monkeypatch.setenv("EVEROS_API_KEY", "test-key")
    monkeypatch.setattr(everos_module, "EverOS", FakeSDK)
    row = {
        "experience_id": "exp-1",
        "payload": {
            "clip_id": "0123456789abcdef",
            "label": "safe_walkway",
            "observations": [
                {"action": "cheap_dense"},
                {"action": "pro_full"},
                {"action": "pro_occluded"},
            ],
            "text": "Best strategy: cheap_single.",
        },
    }
    with EverOSClient() as client:
        assert client.add_experience(row, "seed-session")["status"] == "queued"
        client.flush("seed-session")
        client.search("cheap visual strategy")
        client.get("agent_case")

    calls = FakeSDK.instances[-1].calls
    add = calls[0][1]
    assert add["mode"] == "agent"
    assert add["async_mode"] is False
    assert [message["role"] for message in add["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "tool",
        "assistant",
    ]
    assert add["messages"][1]["sender_id"] == "mavis-agent-v1"
    assert add["messages"][-2]["tool_call_id"] == "evaluate_ground_truth"
    assert add["messages"][-1]["content"].startswith("Task completed successfully.")
    assert calls[2][1]["user_id"] == "mavis-hackathon"
    assert calls[3][1]["memory_type"] == "agent_case"


def test_durable_experience_uses_searchable_episode_track(monkeypatch) -> None:
    monkeypatch.setenv("EVEROS_API_KEY", "test-key")
    monkeypatch.setattr(everos_module, "EverOS", FakeSDK)
    row = {
        "experience_id": "exp-episode",
        "payload": {
            "clip_id": "clip-1",
            "label": "opened_panel_cover",
            "outcome": "unsafe_escalation_rule_verified",
            "best_action": "abstain_and_escalate_unsafe_false_negative",
            "key_insight": "Do not accept a cheap safe prediction.",
            "metadata": {"counterfactual_tested": False},
        },
    }

    with EverOSClient() as client:
        client.add_durable_experience(row, "episode-session")
        client.get_session_episodes("episode-session")
        client.get("episode")

    calls = FakeSDK.instances[-1].calls
    add = calls[0][1]
    assert add["mode"] == "chat"
    assert add["messages"][0]["sender_id"] == "mavis-hackathon"
    assert "ground_truth=opened_panel_cover" in add["messages"][0]["content"]
    assert calls[1][1]["memory_type"] == "episode"
    assert calls[1][1]["filters"] == {"session_id": "episode-session"}
    assert calls[2][1]["user_id"] == "mavis-hackathon"
