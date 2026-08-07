"""EverOS-backed memory.

EverOS stores each inference step as an agent trajectory (``mode="agent"``) and
serves them back through hybrid search, along with the derived agent cases and
skills. Those derived structures are the reason to use it over a flat store: a
case is EverOS's own generalisation over many episodes, and its relevance,
confidence and maturity scores are exactly the weights
:func:`mavis.memory.base.aggregate` wants.

The exact request/response shapes are pinned in ``ENDPOINTS`` and
:meth:`_hits_from_payload`. The field names are read defensively — a missing key
degrades that hit's weight rather than raising — so a schema mismatch shows up as
"memory is not helping" in the benchmark instead of a crashed run. Every failure
falls through to the local store, so a MAVIS run never dies because memory is
down.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from ..types import Action, Episode, MemoryHit, SceneSummary
from .local import LocalMemory

log = logging.getLogger(__name__)

ENDPOINTS = {
    "search": "/api/v2/memory/search",
    "write": "/api/v2/memory",
}


class EverOSMemory:
    """HTTP client for EverOS with a local mirror as the failure path."""

    name = "everos"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        namespace: str = "mavis",
        timeout: float = 6.0,
        mirror: LocalMemory | None = None,
    ):
        self.base_url = (base_url or os.environ.get("EVEROS_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("EVEROS_API_KEY", "")
        if not self.base_url:
            raise RuntimeError(
                "EVEROS_BASE_URL is not set. Run with --memory local to use the "
                "file-backed store instead."
            )
        self.namespace = namespace
        self.timeout = timeout
        # Everything written to EverOS is also written locally: the benchmark
        # must stay reproducible offline, and it gives us the ablation control.
        self.mirror = mirror if mirror is not None else LocalMemory("data/memory/everos_mirror.jsonl")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"
        self.degraded = False

    # -- MemoryStore -------------------------------------------------------

    def search(self, scene: SceneSummary, k: int) -> list[MemoryHit]:
        body = {
            "namespace": self.namespace,
            "mode": "hybrid",
            "query": self._query_text(scene),
            "limit": k,
            "filters": {"kind": "mavis_inference_step"},
        }
        try:
            resp = self.session.post(
                self.base_url + ENDPOINTS["search"], json=body, timeout=self.timeout
            )
            resp.raise_for_status()
            hits = self._hits_from_payload(resp.json(), scene)
            if hits:
                self.degraded = False
                return hits
        except (requests.RequestException, ValueError) as exc:
            if not self.degraded:
                log.warning("EverOS search failed, falling back to mirror: %s", exc)
                self.degraded = True
        return self.mirror.search(scene, k)

    def write(self, episode: Episode) -> None:
        self.mirror.write(episode)
        payload = {
            "namespace": self.namespace,
            "mode": "agent",
            "kind": "mavis_inference_step",
            "text": self._episode_text(episode),
            "metadata": episode.to_dict(),
        }
        try:
            resp = self.session.post(
                self.base_url + ENDPOINTS["write"], json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            if not self.degraded:
                log.warning("EverOS write failed, mirrored locally only: %s", exc)
                self.degraded = True

    def flush(self) -> None:
        self.mirror.flush()

    def __len__(self) -> int:
        return len(self.mirror)

    # -- serialisation -----------------------------------------------------

    @staticmethod
    def _query_text(scene: SceneSummary) -> str:
        parts = list(scene.labels) + list(scene.objects)
        return f"factory CCTV scene: {', '.join(parts)} (cheap risk {scene.risk:.2f})"

    @staticmethod
    def _episode_text(ep: Episode) -> str:
        return (
            f"scene: {', '.join(ep.scene.labels)}; objects: {', '.join(ep.scene.objects)}; "
            f"action: {ep.action.value}; information gain {ep.observed_ig:.3f} bits "
            f"for cost {ep.observed_cost:.6f}; hazard "
            f"{'confirmed' if ep.hazard_outcome else 'ruled out'}"
        )

    def _hits_from_payload(self, payload: Any, scene: SceneSummary) -> list[MemoryHit]:
        """Read EverOS results defensively into MemoryHits.

        Accepts either a bare list or the common ``{"results": [...]}`` /
        ``{"episodes": [...]}`` envelopes, and looks for the measured numbers in
        the metadata block that :meth:`write` produced.
        """
        rows = payload
        if isinstance(payload, dict):
            for key in ("results", "episodes", "memories", "data", "items"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
        if not isinstance(rows, list):
            return []

        hits: list[MemoryHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            meta = row.get("metadata") or row.get("meta") or row
            action_name = meta.get("action")
            if action_name not in Action.__members__ and action_name not in {
                a.value for a in Action
            }:
                continue
            try:
                hits.append(
                    MemoryHit(
                        action=Action(action_name),
                        observed_ig=float(meta.get("observed_ig", 0.0)),
                        observed_cost=float(meta.get("observed_cost", 0.0)),
                        hazard_outcome=bool(meta.get("hazard_outcome", False)),
                        relevance=_first_float(row, ("relevance", "score", "similarity"), 0.5),
                        confidence=_first_float(row, ("confidence",), 0.6),
                        maturity=_first_float(row, ("maturity",), 0.0),
                        scene_signature=meta.get("scene_signature", ""),
                        source="everos",
                    )
                )
            except (TypeError, ValueError):
                continue
        return hits


def _first_float(row: dict, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return default
