from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

from .labels import normalize_label
from .models import EvaluatedAction, Experience, InferenceResult


def normalized_scores(scores: dict[str, float], labels: tuple[str, ...]) -> dict[str, float]:
    values = {label: max(0.0, float(scores.get(label, 0.0))) for label in labels}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: value / total for label, value in values.items()}


def entropy(scores: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in scores.values() if value > 0)


def information_gain(scores: dict[str, float], labels: tuple[str, ...]) -> float:
    prior_entropy = math.log2(len(labels))
    posterior_entropy = entropy(normalized_scores(scores, labels))
    return max(0.0, prior_entropy - posterior_entropy)


def evaluate_actions(
    label: str,
    results: Iterable[InferenceResult],
    labels: tuple[str, ...],
    alpha: float = 1.0,
    beta: float = 0.15,
    gamma: float = 0.30,
    delta: float = 0.0001,
) -> tuple[EvaluatedAction, ...]:
    evaluated: list[EvaluatedAction] = []
    expected = normalize_label(label)
    result_list = list(results)
    cheap_result = next(
        (result for result in result_list if result.action == "cheap_single" and not result.error),
        None,
    )
    cheap_entropy = (
        entropy(normalized_scores(cheap_result.scores, labels))
        if cheap_result is not None
        else None
    )
    for result in result_list:
        correct = normalize_label(result.prediction) == expected and result.error is None
        if result.action == "cheap_single" or cheap_entropy is None:
            gain = information_gain(result.scores, labels)
        else:
            posterior = entropy(normalized_scores(result.scores, labels))
            gain = max(0.0, cheap_entropy - posterior)
        finite_cost = result.cost if math.isfinite(result.cost) else 0.0
        reward = (
            alpha * float(correct) + beta * gain - gamma * finite_cost - delta * result.latency_ms
        )
        evaluated.append(
            EvaluatedAction(result=result, correct=correct, information_gain=gain, reward=reward)
        )
    return tuple(evaluated)


def _best_action(observations: tuple[EvaluatedAction, ...]) -> EvaluatedAction:
    successes = [item for item in observations if item.correct]
    if successes:
        return min(successes, key=lambda item: (item.result.cost, item.result.latency_ms))
    return max(observations, key=lambda item: item.reward)


def _insight(best: EvaluatedAction, observations: tuple[EvaluatedAction, ...]) -> str:
    action = best.result.action
    if not best.correct:
        return "No tested strategy was correct; retain the episode as a hard case and escalate."
    by_action = {item.result.action: item for item in observations}
    if action == "cheap_single":
        return "A cheap single keyframe was sufficient; stronger visual reasoning added cost only."
    strong_multi = by_action.get("strong_multi")
    if action == "strong_single" and strong_multi is not None and strong_multi.correct:
        ratio = strong_multi.result.cost / max(best.result.cost, 1e-12)
        return f"One detailed keyframe was sufficient; temporal reasoning cost {ratio:.2f}x more."
    if action == "crop_strong":
        return "The interaction-region crop preserved the decision while avoiding full-frame cost."
    if action == "strong_multi":
        return "Temporal comparison was required to resolve the safety behavior."
    return f"{action} was the minimum-cost successful strategy."


def build_experience(
    clip_id: str,
    label: str,
    split: str,
    observations: tuple[EvaluatedAction, ...],
    prompt_version: str,
) -> Experience:
    if not observations:
        raise ValueError("Cannot build an experience without observations")
    best = _best_action(observations)
    scene = next((item.result.scene for item in observations if item.result.scene), "factory scene")
    outcome = "success" if best.correct else "failure"
    insight = _insight(best, observations)
    compact = [
        {
            "action": item.result.action,
            "correct": item.correct,
            "cost_credits": None if not math.isfinite(item.result.cost) else item.result.cost,
            "information_gain_proxy": round(item.information_gain, 6),
        }
        for item in observations
    ]
    text = (
        "Task: detect factory safety behavior.\n"
        f"Scene: {scene}.\n"
        f"Ground truth: {label}.\n"
        f"Strategy trials: {json.dumps(compact, separators=(',', ':'))}.\n"
        f"Best strategy: {best.result.action}.\n"
        f"Key insight: {insight}\n"
        f"Outcome: {outcome}."
    )
    identity = f"{clip_id}:{prompt_version}:{','.join(item.result.action for item in observations)}"
    experience_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return Experience(
        experience_id=experience_id,
        clip_id=clip_id,
        label=label,
        split=split,
        scene=scene,
        observations=observations,
        best_action=best.result.action,
        key_insight=insight,
        outcome=outcome,
        text=text,
        metadata={
            "prompt_version": prompt_version,
            "information_gain_is_proxy": True,
            "cost_source": "actual_credits_when_present_else_estimate",
        },
    )
