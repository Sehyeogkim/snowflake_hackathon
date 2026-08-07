"""JSONL-backed memory store.

Serves two purposes: it is the fallback when EverOS is unreachable, and it is the
control arm of the ablation — running MAVIS with an empty ``LocalMemory`` shows
what the scheduler is worth with no prior experience at all.

Storage is a plain append-only JSONL file because episodes are small, the corpus
is thousands of rows at most, and a reviewer should be able to read the memory
with ``less``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Action, Episode, MemoryHit, SceneSummary
from .base import scene_similarity


class LocalMemory:
    """In-process episode store with brute-force similarity search."""

    name = "local"

    def __init__(self, path: str | Path | None = "data/memory/episodes.jsonl", *, load: bool = True):
        self.path = Path(path) if path else None
        self._episodes: list[Episode] = []
        if load and self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._episodes.append(_episode_from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # a truncated tail must not brick the store

    # -- MemoryStore -------------------------------------------------------

    def search(self, scene: SceneSummary, k: int) -> list[MemoryHit]:
        if not self._episodes:
            return []
        scored = []
        for ep in self._episodes:
            rel = scene_similarity(scene, ep.scene)
            if rel <= 0.05:
                continue
            scored.append((rel, ep))
        scored.sort(key=lambda t: t[0], reverse=True)

        hits: list[MemoryHit] = []
        for rel, ep in scored[:k]:
            hits.append(
                MemoryHit(
                    action=ep.action,
                    observed_ig=ep.observed_ig,
                    observed_cost=ep.observed_cost,
                    hazard_outcome=ep.hazard_outcome,
                    relevance=rel,
                    confidence=0.75,
                    maturity=min(len(self._episodes) / 200.0, 1.0),
                    scene_signature=ep.scene.signature(),
                    source="local",
                )
            )
        return hits

    def write(self, episode: Episode) -> None:
        self._episodes.append(episode)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    def flush(self) -> None:  # append-on-write, nothing buffered
        pass

    def __len__(self) -> int:
        return len(self._episodes)


def _episode_from_dict(d: dict) -> Episode:
    scene = d["scene"]
    return Episode(
        scene=SceneSummary(
            labels=scene.get("labels", []),
            objects=scene.get("objects", []),
            risk=float(scene.get("risk", 0.0)),
        ),
        action=Action(d["action"]),
        observed_ig=float(d["observed_ig"]),
        observed_cost=float(d["observed_cost"]),
        hazard_outcome=bool(d["hazard_outcome"]),
        hazard_prob_before=float(d.get("hazard_prob_before", 0.0)),
        hazard_prob_after=float(d.get("hazard_prob_after", 0.0)),
        clip_id=d.get("clip_id", ""),
        timestamp_s=float(d.get("timestamp_s", 0.0)),
    )
