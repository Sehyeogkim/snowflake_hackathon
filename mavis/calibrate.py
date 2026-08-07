"""Measure what each action actually costs on the backend in use.

The scheduler's cold-start priors are the numbers it trades against before any
memory exists, and a wrong one distorts every early decision. Hand-written
constants cannot survive a backend swap: values invented for Snowflake credits
overpriced Gemini's cheap tier by roughly thirteen times, which made MAVIS skip
cheap looks that were effectively free and cost it recall.

So the priors are measured, not assumed. Three calls on a representative frame —
one per tier — are enough, and they matter because prompt tokens are dominated by
the image, so the frame's resolution, not the prompt text, sets the price.

Calibration calls are billed like any other. They are made once per run and their
cost is reported separately rather than folded into either policy's total, since
neither policy made them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config
from .cortex.base import HAZARD_PROMPT, MULTIFRAME_PROMPT
from .dataset import CLASSIFY_CATEGORIES
from .types import Action, CostRecord

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Calibration:
    costs: dict[Action, float]
    records: list[CostRecord]

    @property
    def total_cost(self) -> float:
        return sum(r.credits or 0.0 for r in self.records)

    def describe(self) -> str:
        rows = "  ".join(f"{a.value}={c:.7f}" for a, c in sorted(self.costs.items(), key=lambda t: t[0].value))
        return f"calibrated action costs: {rows}  (spent {self.total_cost:.6f} to measure)"


def calibrate(cortex, cfg: Config, frame: np.ndarray) -> Calibration | None:
    """Measure per-action cost on ``frame`` and write it into ``cfg``.

    Returns None for backends that report no cost at call time (the Snowflake
    path defers to usage history), leaving the static priors in place.
    """
    records: list[CostRecord] = []

    _scene, c_classify = cortex.classify(frame, CLASSIFY_CATEGORIES)
    records.append(c_classify)
    _r, c_cheap = cortex.complete([frame], HAZARD_PROMPT, strong=False)
    records.append(c_cheap)
    _r, c_strong = cortex.complete([frame], HAZARD_PROMPT, strong=True)
    records.append(c_strong)

    if any(r.credits is None for r in records):
        log.info("backend does not report cost at call time; keeping static priors")
        return None

    strong = c_strong.credits or 0.0
    costs = {
        Action.SKIP: 0.0,
        Action.CLASSIFY: c_classify.credits or 0.0,
        Action.CHEAP_VLM: c_cheap.credits or 0.0,
        Action.STRONG_VLM: strong,
        # Two images roughly doubles the prompt half of a strong call; the
        # completion half is unchanged, so the multiplier is under 2.
        Action.MULTIFRAME: strong * 1.8,
    }
    cfg.scheduler.prior_cost = costs
    calibration = Calibration(costs=costs, records=records)
    log.info("%s", calibration.describe())
    return calibration


def sample_frame(clips, cfg: Config) -> np.ndarray | None:
    """First gated frame of the first readable clip — a real frame, not a swatch.

    Image tokens dominate the bill and scale with resolution, so calibrating on a
    synthetic image of the wrong size would reproduce the very error this module
    exists to prevent.
    """
    from .gate import iter_gated_frames

    for clip in clips:
        try:
            for frame, _stats in iter_gated_frames(clip.path, cfg.gate):
                return frame.image
        except RuntimeError:
            continue
    return None
