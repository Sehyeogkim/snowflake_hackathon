"""Cortex clients: the single boundary between MAVIS and any AI inference."""

from __future__ import annotations

from .base import HAZARD_PROMPT, MULTIFRAME_PROMPT, CortexClient, CostLedger
from .mock import MockCortex

__all__ = [
    "CortexClient",
    "CostLedger",
    "MockCortex",
    "HAZARD_PROMPT",
    "MULTIFRAME_PROMPT",
    "build",
]


KINDS = ("mock", "gemini", "snowflake")


def build(kind: str, *, seed: int = 0, **kwargs) -> CortexClient:
    """Construct a client by name. Real backends are imported lazily.

    Credentials and optional dependencies are only touched when actually
    requested, so the mock path runs on a bare checkout.
    """
    if kind == "mock":
        return MockCortex(seed=seed)
    if kind == "gemini":
        from .gemini import GeminiCortex

        return GeminiCortex(**kwargs)
    if kind == "snowflake":
        from .snowflake import SnowflakeCortex

        return SnowflakeCortex(**kwargs)
    raise ValueError(f"unknown cortex client: {kind!r} (expected one of {KINDS})")
