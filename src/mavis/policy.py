from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .evaluation import entropy, normalized_scores
from .labels import UNSAFE_LABELS


class RuntimeAction(StrEnum):
    ACCEPT_CHEAP = "accept_cheap"
    STRONG_SINGLE = "strong_single"
    STRONG_MULTI = "strong_multi"


@dataclass(frozen=True)
class MemoryEvidence:
    action: RuntimeAction
    retrieval_weight: float
    correct: bool
    information_gain: float
    cost_credits: float
    source_case_id: str | None = None
    unsafe_false_negative: bool = False
    recommends_escalation: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    cheap_accept_confidence: float = 0.86
    cheap_accept_max_risk: float = 0.20
    safety_override_risk: float = 0.65
    safety_override_entropy: float = 2.35
    minimum_empirical_success: float = 0.65
    minimum_support: float = 1.5
    risk_weight: float = 0.75
    unsafe_memory_minimum_support: float = 0.35
    unsafe_memory_failure_rate: float = 0.40
    safe_accept_minimum_support: float = 0.80
    safe_accept_minimum_success: float = 0.72


@dataclass(frozen=True)
class PolicyDecision:
    action: RuntimeAction
    reason: str
    utilities: dict[str, float]
    entropy: float
    confidence: float


class MavisPolicy:
    """Safety-gated memory-conditioned information gain per Cortex credit."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    @staticmethod
    def _posterior_success(items: list[MemoryEvidence]) -> tuple[float, float]:
        # Beta(1,1) shrinkage avoids trusting one lucky retrieved episode.
        support = sum(max(0.0, item.retrieval_weight) for item in items)
        successes = sum(max(0.0, item.retrieval_weight) for item in items if item.correct)
        return (successes + 1.0) / (support + 2.0), support

    @staticmethod
    def _weighted_mean(items: list[MemoryEvidence], field: str, fallback: float) -> float:
        weights = [max(0.0, item.retrieval_weight) for item in items]
        total = sum(weights)
        if total <= 0:
            return fallback
        return (
            sum(weight * float(getattr(item, field)) for weight, item in zip(weights, items))
            / total
        )

    def decide(
        self,
        cheap_scores: dict[str, float],
        labels: tuple[str, ...],
        risk: float,
        need_temporal_context: bool,
        evidence: list[MemoryEvidence],
        fallback_costs: dict[RuntimeAction, float] | None = None,
        unsafe_labels: frozenset[str] = UNSAFE_LABELS,
    ) -> PolicyDecision:
        scores = normalized_scores(cheap_scores, labels)
        uncertainty = entropy(scores)
        confidence = max(scores.values())
        prediction = max(scores, key=scores.get)
        cfg = self.config

        # A confidently wrong "safe" answer is the dominant failure mode in the
        # dense seed run. Retrieved, GT-evaluated false negatives therefore act
        # as a hard routing guardrail before model confidence is considered.
        unsafe_memory = [
            item
            for item in evidence
            if item.unsafe_false_negative or item.recommends_escalation
        ]
        unsafe_support = sum(max(0.0, item.retrieval_weight) for item in evidence)
        unsafe_failures = sum(
            max(0.0, item.retrieval_weight)
            for item in unsafe_memory
            if item.unsafe_false_negative
        )
        unsafe_failure_rate = unsafe_failures / max(unsafe_support, 1e-9)
        if (
            prediction not in unsafe_labels
            and unsafe_support >= cfg.unsafe_memory_minimum_support
            and unsafe_failure_rate >= cfg.unsafe_memory_failure_rate
        ):
            return PolicyDecision(
                action=RuntimeAction.STRONG_MULTI,
                reason="memory guardrail: similar GT episodes contain unsafe false negatives",
                utilities={RuntimeAction.STRONG_MULTI.value: 1e12},
                entropy=uncertainty,
                confidence=confidence,
            )

        if (
            risk >= cfg.safety_override_risk
            or uncertainty >= cfg.safety_override_entropy
            or need_temporal_context
        ):
            return PolicyDecision(
                action=RuntimeAction.STRONG_MULTI,
                reason="safety override: high risk or uncertainty",
                utilities={},
                entropy=uncertainty,
                confidence=confidence,
            )
        cheap_success, cheap_support = self._posterior_success(
            [item for item in evidence if item.action == RuntimeAction.ACCEPT_CHEAP]
        )
        safe_accept_supported = (
            prediction in unsafe_labels
            or (
                cheap_support >= cfg.safe_accept_minimum_support
                and cheap_success >= cfg.safe_accept_minimum_success
            )
        )
        if (
            confidence >= cfg.cheap_accept_confidence
            and risk <= cfg.cheap_accept_max_risk
            and safe_accept_supported
        ):
            return PolicyDecision(
                action=RuntimeAction.ACCEPT_CHEAP,
                reason="cheap result cleared the confidence and low-risk gates",
                utilities={RuntimeAction.ACCEPT_CHEAP.value: 1e12},
                entropy=uncertainty,
                confidence=confidence,
            )
        if confidence >= cfg.cheap_accept_confidence and risk <= cfg.cheap_accept_max_risk:
            return PolicyDecision(
                action=RuntimeAction.STRONG_SINGLE,
                reason="unverified safe result: confirm with one strong keyframe",
                utilities={RuntimeAction.STRONG_SINGLE.value: 1e12},
                entropy=uncertainty,
                confidence=confidence,
            )

        defaults = fallback_costs or {
            RuntimeAction.STRONG_SINGLE: 1.0,
            RuntimeAction.STRONG_MULTI: 2.2,
        }
        utilities: dict[str, float] = {}
        feasible: list[RuntimeAction] = []
        for action, fallback_gain_ratio in (
            (RuntimeAction.STRONG_SINGLE, 0.45),
            (RuntimeAction.STRONG_MULTI, 0.72),
        ):
            items = [item for item in evidence if item.action == action]
            success, support = self._posterior_success(items)
            gain = self._weighted_mean(items, "information_gain", uncertainty * fallback_gain_ratio)
            cost = self._weighted_mean(items, "cost_credits", defaults[action])
            if action == RuntimeAction.STRONG_MULTI and need_temporal_context:
                gain *= 1.35
            # Retrieved correctness is a feasibility gate, not an IG bonus. The
            # conservative multi-frame action remains available as the fallback.
            if (
                support >= cfg.minimum_support and success >= cfg.minimum_empirical_success
            ) or action == RuntimeAction.STRONG_MULTI:
                feasible.append(action)
            memory_support = min(1.0, support / max(cfg.minimum_support, 1e-9))
            risk_adjusted_gain = gain * (1.0 + cfg.risk_weight * max(0.0, risk))
            utilities[action.value] = (risk_adjusted_gain + 0.10 * memory_support) / max(cost, 1e-9)

        selected = max(feasible, key=lambda action: utilities[action.value])
        return PolicyDecision(
            action=selected,
            reason="highest feasible memory-conditioned information gain per billed credit",
            utilities=utilities,
            entropy=uncertainty,
            confidence=confidence,
        )
