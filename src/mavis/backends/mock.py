from __future__ import annotations

import hashlib
import random
import time
import uuid

from ..labels import CANONICAL_LABELS
from ..models import CandidateSet, ClipRecord, InferenceResult
from .base import InferenceBackend

_ACTION_PROFILE = {
    "cheap_single": {"accuracy": 0.68, "cost": 0.08, "latency": 120.0},
    "strong_single": {"accuracy": 0.88, "cost": 0.34, "latency": 420.0},
    "strong_multi": {"accuracy": 0.95, "cost": 0.82, "latency": 760.0},
    "crop_strong": {"accuracy": 0.90, "cost": 0.29, "latency": 390.0},
}


class MockInferenceBackend(InferenceBackend):
    """Deterministic local backend for pipeline validation; never a benchmark result."""

    evidence_source = "mock"

    def __init__(self, cheap_model: str, strong_model: str, seed: int = 42) -> None:
        self.cheap_model = cheap_model
        self.strong_model = strong_model
        self.seed = seed

    def infer(self, clip: ClipRecord, candidates: CandidateSet, action: str) -> InferenceResult:
        del candidates
        profile = _ACTION_PROFILE[action]
        digest = hashlib.sha256(f"{self.seed}:{clip.clip_id}:{action}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        correct = rng.random() < profile["accuracy"]
        alternatives = [label for label in CANONICAL_LABELS if label != clip.label]
        prediction = clip.label if correct else rng.choice(alternatives)
        confidence = rng.uniform(0.72, 0.96) if correct else rng.uniform(0.40, 0.72)
        residual = (1.0 - confidence) / (len(CANONICAL_LABELS) - 1)
        scores = {label: residual for label in CANONICAL_LABELS}
        scores[prediction] = confidence
        latency = profile["latency"] * rng.uniform(0.85, 1.15)
        time.sleep(min(latency / 1000.0, 0.015))
        return InferenceResult(
            action=action,
            model=self.cheap_model if action == "cheap_single" else self.strong_model,
            prediction=prediction,
            scores=scores,
            # Keep the structural backend deterministic without leaking the GT
            # label into the memory query used by the runtime scheduler.
            scene="factory floor with a worker, marked areas, equipment, and a moving action",
            risk=sum(
                scores[label]
                for label in scores
                if label
                in {
                    "safe_walkway_violation",
                    "unauthorized_intervention",
                    "opened_panel_cover",
                    "carrying_overload_with_forklift",
                }
            ),
            need_temporal_context=action == "cheap_single" and confidence < 0.65,
            query_id=f"mock-{uuid.uuid4()}",
            latency_ms=latency,
            estimated_credits=float(profile["cost"]),
            actual_credits=float(profile["cost"]),
            raw_response={"mock": True, "warning": "not valid for cost-reduction claims"},
        )
