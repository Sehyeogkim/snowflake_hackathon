"""MAVIS — Memory-Aware Visual Inference Scheduler.

Not every frame deserves intelligence. Remember what was useful; reason only
when it is worth the cost.

    Snowflake  sees the frame and reports what it cost.
    EverOS     remembers which observations paid for themselves.
    MAVIS      decides whether the next one is worth paying for.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, DEFAULT
from .types import Action, Decision, Episode, SceneSummary, Step, Trace

__all__ = [
    "Action",
    "Config",
    "DEFAULT",
    "Decision",
    "Episode",
    "SceneSummary",
    "Step",
    "Trace",
    "__version__",
]
