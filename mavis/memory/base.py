"""The memory boundary.

MAVIS asks memory exactly one question: *for a scene like this one, which
observations have historically paid for themselves?* Everything else — how
episodes are embedded, whether they live in EverOS or a local file — is behind
this protocol.

The answer is a list of :class:`~mavis.types.MemoryHit`, each carrying the
information gain that was actually measured and the cost that was actually paid.
Not a prediction someone made at the time: the realised numbers.
"""

from __future__ import annotations

from typing import Protocol

from ..types import Action, Episode, MemoryHit, SceneSummary


class MemoryStore(Protocol):
    name: str

    def search(self, scene: SceneSummary, k: int) -> list[MemoryHit]:
        """Recall the k most relevant past episodes for this scene."""
        ...

    def write(self, episode: Episode) -> None:
        """Persist one (scene, action, outcome) step."""
        ...

    def flush(self) -> None: ...

    def __len__(self) -> int: ...


def scene_similarity(a: SceneSummary, b: SceneSummary) -> float:
    """Relevance between two cheap scene descriptions, in [0, 1].

    Deliberately blunt and dependency-free: Jaccard overlap on labels and
    objects, plus agreement on the cheap risk score. The cheap tier is the only
    thing available *before* deciding whether to pay for the expensive tier, so
    similarity must be computable from it alone.
    """
    def jaccard(x: list[str], y: list[str]) -> float:
        sx, sy = {s.lower() for s in x}, {s.lower() for s in y}
        if not sx and not sy:
            return 0.0
        return len(sx & sy) / len(sx | sy)

    label_sim = jaccard(a.labels, b.labels)
    object_sim = jaccard(a.objects, b.objects)
    risk_sim = 1.0 - min(abs(a.risk - b.risk), 1.0)
    return 0.5 * label_sim + 0.3 * object_sim + 0.2 * risk_sim


def aggregate(hits: list[MemoryHit], action: Action) -> tuple[float, float, float]:
    """Collapse recalled episodes into ``(expected_ig, expected_cost, support)``.

    Each hit is weighted by relevance × confidence, so a barely-related memory
    from a low-confidence run barely moves the estimate. ``support`` is the total
    weight, which the scheduler uses to decide how far to trust memory over its
    static prior — this is what makes cold start degrade gracefully instead of
    producing confident nonsense from a single episode.
    """
    relevant = [h for h in hits if h.action is action]
    if not relevant:
        return 0.0, 0.0, 0.0

    weights = [max(h.relevance, 0.0) * max(h.confidence, 0.05) for h in relevant]
    total = sum(weights)
    if total <= 0:
        return 0.0, 0.0, 0.0

    ig = sum(w * h.observed_ig for w, h in zip(weights, relevant)) / total
    cost = sum(w * h.observed_cost for w, h in zip(weights, relevant)) / total
    return ig, cost, total
