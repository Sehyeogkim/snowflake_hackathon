from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    path: Path
    label: str
    split: str = "unassigned"
    group_id: str | None = None
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["path"] = str(self.path)
        return row


@dataclass(frozen=True)
class CandidateFrame:
    name: str
    frame_index: int
    timestamp_s: float
    path: Path
    motion_score: float = 0.0
    sharpness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["path"] = str(self.path)
        return row


@dataclass(frozen=True)
class CandidateSet:
    clip_id: str
    fps: float
    frame_count: int
    duration_s: float
    early: CandidateFrame
    peak: CandidateFrame
    late: CandidateFrame
    crop: CandidateFrame | None = None

    def for_action(self, action: str) -> list[Path]:
        if action in {"cheap_single", "strong_single"}:
            return [self.peak.path]
        if action == "strong_multi":
            return [self.early.path, self.peak.path, self.late.path]
        if action == "crop_strong" and self.crop:
            return [self.crop.path]
        raise ValueError(f"Unsupported action or missing candidate: {action}")


@dataclass(frozen=True)
class InferenceResult:
    action: str
    model: str
    prediction: str
    scores: dict[str, float]
    scene: str
    risk: float
    need_temporal_context: bool
    query_id: str | None
    latency_ms: float
    estimated_credits: float | None = None
    actual_credits: float | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def cost(self) -> float:
        if self.actual_credits is not None:
            return self.actual_credits
        if self.estimated_credits is not None:
            return self.estimated_credits
        return float("inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatedAction:
    result: InferenceResult
    correct: bool
    information_gain: float
    reward: float

    def to_dict(self) -> dict[str, Any]:
        row = self.result.to_dict()
        row.update(
            correct=self.correct,
            information_gain=self.information_gain,
            reward=self.reward,
        )
        return row


@dataclass(frozen=True)
class Experience:
    experience_id: str
    clip_id: str
    label: str
    split: str
    scene: str
    observations: tuple[EvaluatedAction, ...]
    best_action: str
    key_insight: str
    outcome: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "clip_id": self.clip_id,
            "label": self.label,
            "split": self.split,
            "scene": self.scene,
            "observations": [item.to_dict() for item in self.observations],
            "best_action": self.best_action,
            "key_insight": self.key_insight,
            "outcome": self.outcome,
            "text": self.text,
            "metadata": self.metadata,
        }
