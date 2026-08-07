from mavis.evaluation import build_experience, evaluate_actions, information_gain
from mavis.labels import CANONICAL_LABELS
from mavis.models import InferenceResult


def result(action: str, prediction: str, cost: float, confidence: float) -> InferenceResult:
    residual = (1 - confidence) / 7
    scores = {label: residual for label in CANONICAL_LABELS}
    scores[prediction] = confidence
    return InferenceResult(
        action=action,
        model="test",
        prediction=prediction,
        scores=scores,
        scene="worker near panel",
        risk=0.8,
        need_temporal_context=False,
        query_id=f"q-{action}",
        latency_ms=100,
        actual_credits=cost,
    )


def test_information_gain_increases_with_confidence() -> None:
    low = {label: 1 / 8 for label in CANONICAL_LABELS}
    high = {label: (0.9 if index == 0 else 0.1 / 7) for index, label in enumerate(CANONICAL_LABELS)}
    assert information_gain(high, CANONICAL_LABELS) > information_gain(low, CANONICAL_LABELS)


def test_minimum_cost_correct_action_wins() -> None:
    observations = evaluate_actions(
        "unauthorized_intervention",
        [
            result("cheap_single", "authorized_intervention", 0.1, 0.7),
            result("strong_single", "unauthorized_intervention", 0.4, 0.9),
            result("strong_multi", "unauthorized_intervention", 0.9, 0.95),
        ],
        CANONICAL_LABELS,
    )
    experience = build_experience(
        "clip-1", "unauthorized_intervention", "seed", observations, "test-v1"
    )
    assert experience.best_action == "strong_single"
    assert "temporal reasoning cost" in experience.key_insight.casefold()
