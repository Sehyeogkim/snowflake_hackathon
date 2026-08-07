from mavis.labels import CANONICAL_LABELS
from mavis.policy import MavisPolicy, MemoryEvidence, RuntimeAction


def scores(confidence: float) -> dict[str, float]:
    residual = (1 - confidence) / 7
    values = {label: residual for label in CANONICAL_LABELS}
    values[CANONICAL_LABELS[0]] = confidence
    return values


def test_high_risk_forces_multiframe() -> None:
    decision = MavisPolicy().decide(scores(0.95), CANONICAL_LABELS, 0.9, False, [])
    assert decision.action == RuntimeAction.STRONG_MULTI


def test_low_risk_high_confidence_safe_requires_memory_support() -> None:
    decision = MavisPolicy().decide(scores(0.92), CANONICAL_LABELS, 0.1, False, [])
    assert decision.action == RuntimeAction.STRONG_SINGLE


def test_verified_safe_memory_allows_cheap_accept() -> None:
    evidence = [
        MemoryEvidence(RuntimeAction.ACCEPT_CHEAP, 1.0, True, 0.0, 0.1),
        MemoryEvidence(RuntimeAction.ACCEPT_CHEAP, 1.0, True, 0.0, 0.1),
    ]
    decision = MavisPolicy().decide(scores(0.92), CANONICAL_LABELS, 0.1, False, evidence)
    assert decision.action == RuntimeAction.ACCEPT_CHEAP


def test_retrieved_unsafe_false_negative_forces_multiframe() -> None:
    evidence = [
        MemoryEvidence(
            RuntimeAction.STRONG_MULTI,
            0.8,
            True,
            1.0,
            1.0,
            unsafe_false_negative=True,
            recommends_escalation=True,
        )
    ]
    decision = MavisPolicy().decide(scores(0.95), CANONICAL_LABELS, 0.1, False, evidence)
    assert decision.action == RuntimeAction.STRONG_MULTI
    assert "false negatives" in decision.reason


def test_supported_single_can_beat_expensive_multi() -> None:
    evidence = [
        MemoryEvidence(RuntimeAction.STRONG_SINGLE, 1.0, True, 0.8, 0.3),
        MemoryEvidence(RuntimeAction.STRONG_SINGLE, 1.0, True, 0.7, 0.3),
        MemoryEvidence(RuntimeAction.STRONG_MULTI, 1.0, True, 0.9, 1.0),
    ]
    decision = MavisPolicy().decide(scores(0.7), CANONICAL_LABELS, 0.4, False, evidence)
    assert decision.action == RuntimeAction.STRONG_SINGLE
