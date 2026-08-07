from mavis.memory_evidence import (
    PortableExperienceIndex,
    episode_references,
    evidence_from_experience_payload,
)
from mavis.policy import RuntimeAction


def test_episode_reference_parses_current_canonical_session() -> None:
    response = {
        "episodes": [
            {
                "id": "episode-1",
                "session_id": "mavis-memory-v3-199149e754e4e14546a97f1a",
                "score": 0.62,
            }
        ]
    }
    assert episode_references(response) == [
        ("episode-1", "199149e754e4e14546a97f1a", 0.62)
    ]


def test_unsafe_teacher_episode_becomes_hard_guardrail() -> None:
    payload = {
        "outcome": "unsafe_escalation_rule_verified",
        "best_action": "abstain_and_escalate_unsafe_false_negative",
        "task_completed": True,
        "observations": [{"unsafe_false_negative": True}],
        "metadata": {"unsafe_false_negative": True},
    }
    evidence = evidence_from_experience_payload(payload, 0.7, "episode-1")
    assert evidence[0].action == RuntimeAction.STRONG_MULTI
    assert evidence[0].unsafe_false_negative
    assert evidence[0].recommends_escalation


def test_local_index_returns_auditable_payload_evidence() -> None:
    row = {
        "experience_id": "a" * 24,
        "payload": {
            "scene": "worker walks outside the marked factory walkway",
            "label": "safe_walkway_violation",
            "outcome": "unsafe_escalation_rule_verified",
            "best_action": "abstain_and_escalate_unsafe_false_negative",
            "task_completed": True,
            "observations": [{"unsafe_false_negative": True}],
            "metadata": {"unsafe_false_negative": True},
        },
    }
    index = PortableExperienceIndex([row])
    references = index.local_references("worker outside marked walkway", top_k=1)
    evidence = index.evidence(references)
    assert references[0][1] == "a" * 24
    assert evidence[0].source_case_id == f"local:{'a' * 24}"
