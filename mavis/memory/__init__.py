"""Memory stores: what MAVIS remembers about which observations paid off."""

from __future__ import annotations

import logging

from .base import MemoryStore, aggregate, scene_similarity
from .local import LocalMemory

log = logging.getLogger(__name__)

__all__ = ["MemoryStore", "LocalMemory", "aggregate", "scene_similarity", "build"]


def build(kind: str, **kwargs) -> MemoryStore:
    """Construct a store by name.

    ``everos`` degrades to ``local`` if the service is not configured, because a
    missing memory backend should weaken the result, not stop the run.
    """
    if kind == "local":
        return LocalMemory(**kwargs)
    if kind == "none":
        return LocalMemory(path=None, load=False)
    if kind == "everos":
        from .everos import EverOSMemory

        try:
            return EverOSMemory(**kwargs)
        except RuntimeError as exc:
            log.warning("EverOS unavailable (%s); using local memory", exc)
            return LocalMemory()
    raise ValueError(f"unknown memory store: {kind!r} (expected 'everos', 'local' or 'none')")
