from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import CandidateSet, ClipRecord, InferenceResult


class InferenceBackend(ABC):
    evidence_source = "unknown"

    @abstractmethod
    def infer(self, clip: ClipRecord, candidates: CandidateSet, action: str) -> InferenceResult:
        """Run one visual inference action and return telemetry."""

    def reconcile_costs(self, query_ids: list[str], wait_seconds: int = 0) -> dict[str, float]:
        """Return actual credits keyed by query ID when the provider exposes them."""
        del query_ids, wait_seconds
        return {}

    def close(self) -> None:
        pass
