"""Core value types shared across the MAVIS pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Action(str, Enum):
    """What MAVIS can choose to do with a frame that survived the cheap gate."""

    SKIP = "SKIP"
    CLASSIFY = "CLASSIFY"
    CHEAP_VLM = "CHEAP_VLM"
    STRONG_VLM = "STRONG_VLM"
    MULTIFRAME = "MULTIFRAME"

    @property
    def calls_cortex(self) -> bool:
        return self is not Action.SKIP


#: Actions ordered cheapest-first. Used for tie-breaking and for priors.
ACTION_ORDER = [
    Action.SKIP,
    Action.CLASSIFY,
    Action.CHEAP_VLM,
    Action.STRONG_VLM,
    Action.MULTIFRAME,
]


@dataclass(slots=True)
class Frame:
    """A single decoded frame that the cheap gate let through."""

    index: int
    timestamp_s: float
    image: Any = field(repr=False, default=None)  # np.ndarray, kept out of repr
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "timestamp_s": self.timestamp_s, "path": self.path}


@dataclass(slots=True)
class CostRecord:
    """Actual measured cost of one Cortex call.

    Token counts come from ``AI_COMPLETE(..., show_details => TRUE)`` at call
    time. ``credits`` is left None until reconciled against
    ``CORTEX_AI_FUNCTIONS_USAGE_HISTORY``, which lags by up to a few hours —
    see :mod:`mavis.usage`.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    credits: float | None = None
    query_id: str | None = None
    latency_s: float = 0.0
    estimated: bool = False  # True when produced by the mock client

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


ZERO_COST = CostRecord(model="none")


@dataclass(slots=True)
class SceneSummary:
    """Cheap visual understanding: the output of AI_CLASSIFY on one frame."""

    labels: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    risk: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """Stable, human-readable key for logs and for memory lookup."""
        return "|".join(sorted(self.labels)) + "//" + "|".join(sorted(self.objects))

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "objects": self.objects,
            "risk": self.risk,
        }


@dataclass(slots=True)
class InferenceResult:
    """Output of a VLM reasoning call (cheap or strong)."""

    hazard_prob: float
    confidence: float = 0.5
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_prob": self.hazard_prob,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class MemoryHit:
    """One recalled episode: what was observed, and whether it paid off."""

    action: Action
    observed_ig: float
    observed_cost: float
    hazard_outcome: bool
    relevance: float = 0.0
    confidence: float = 0.5
    maturity: float = 0.0
    scene_signature: str = ""
    source: str = "local"

    @property
    def utility(self) -> float:
        """Information gain per unit cost — the thing worth remembering."""
        return self.observed_ig / self.observed_cost if self.observed_cost > 0 else 0.0


@dataclass(slots=True)
class Episode:
    """A single (scene, action, outcome) trajectory step written back to memory."""

    scene: SceneSummary
    action: Action
    observed_ig: float
    observed_cost: float
    hazard_outcome: bool
    hazard_prob_before: float
    hazard_prob_after: float
    clip_id: str = ""
    timestamp_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "timestamp_s": self.timestamp_s,
            "scene": self.scene.to_dict(),
            "scene_signature": self.scene.signature(),
            "action": self.action.value,
            "observed_ig": self.observed_ig,
            "observed_cost": self.observed_cost,
            "hazard_outcome": self.hazard_outcome,
            "hazard_prob_before": self.hazard_prob_before,
            "hazard_prob_after": self.hazard_prob_after,
        }


@dataclass(slots=True)
class Decision:
    """What MAVIS chose for one frame, and why."""

    action: Action
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    expected_ig: float = 0.0
    expected_cost: float = 0.0
    memory_hits: int = 0
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "scores": self.scores,
            "expected_ig": self.expected_ig,
            "expected_cost": self.expected_cost,
            "memory_hits": self.memory_hits,
            "forced": self.forced,
        }


@dataclass(slots=True)
class Step:
    """One frame's worth of pipeline activity, for the trace and the demo overlay."""

    frame_index: int
    timestamp_s: float
    decision: Decision
    scene: SceneSummary | None = None
    inference: InferenceResult | None = None
    cost: CostRecord | None = None
    hazard_prob_before: float = 0.0
    hazard_prob_after: float = 0.0
    observed_ig: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "decision": self.decision.to_dict(),
            "scene": self.scene.to_dict() if self.scene else None,
            "inference": self.inference.to_dict() if self.inference else None,
            "cost": self.cost.to_dict() if self.cost else None,
            "hazard_prob_before": self.hazard_prob_before,
            "hazard_prob_after": self.hazard_prob_after,
            "observed_ig": self.observed_ig,
        }


@dataclass(slots=True)
class Trace:
    """Everything that happened while one policy processed one clip."""

    clip_id: str
    policy: str
    label_hazard: bool
    label_class: str
    duration_s: float
    frames_decoded: int
    frames_gated: int
    steps: list[Step] = field(default_factory=list)

    # ---- derived ---------------------------------------------------------

    def calls(self, action: Action) -> int:
        return sum(1 for s in self.steps if s.decision.action is action)

    @property
    def cortex_calls(self) -> int:
        return sum(1 for s in self.steps if s.cost is not None)

    @property
    def total_tokens(self) -> int:
        return sum(s.cost.total_tokens for s in self.steps if s.cost)

    @property
    def total_credits(self) -> float:
        return sum(s.cost.credits or 0.0 for s in self.steps if s.cost)

    def first_crossing(self, threshold: float) -> float | None:
        """Timestamp at which hazard belief first exceeded ``threshold``."""
        for s in self.steps:
            if s.hazard_prob_after >= threshold:
                return s.timestamp_s
        return None

    def peak_hazard_prob(self) -> float:
        return max((s.hazard_prob_after for s in self.steps), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "policy": self.policy,
            "label_hazard": self.label_hazard,
            "label_class": self.label_class,
            "duration_s": self.duration_s,
            "frames_decoded": self.frames_decoded,
            "frames_gated": self.frames_gated,
            "cortex_calls": self.cortex_calls,
            "total_tokens": self.total_tokens,
            "total_credits": self.total_credits,
            "peak_hazard_prob": self.peak_hazard_prob(),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
