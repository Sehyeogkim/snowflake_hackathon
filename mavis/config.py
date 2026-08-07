"""Tunable constants for the MAVIS pipeline.

Everything a reviewer might want to challenge lives here rather than being
scattered through the code, so a benchmark run can be reproduced from one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Action


@dataclass(slots=True)
class GateConfig:
    """OpenCV cheap gate: drop duplicate and no-motion frames before any AI call."""

    #: Decode every Nth frame. At 30fps, 5 gives 6 candidate frames per second.
    decode_stride: int = 5
    #: Downscale width used for motion/dHash comparison (not for inference).
    work_width: int = 320
    #: Mean absolute frame difference below this counts as "no motion".
    motion_threshold: float = 2.0
    #: Perceptual-hash Hamming distance below this counts as a duplicate.
    dhash_threshold: int = 6
    #: Always let a frame through if this long has passed, regardless of motion.
    max_gap_s: float = 3.0


@dataclass(slots=True)
class BeliefConfig:
    """Hazard belief state, maintained in log-odds."""

    prior: float = 0.12
    #: Per-second pull of the belief back toward the prior when nothing is observed.
    decay_per_s: float = 0.15
    #: How much each action's observation moves the log-odds.
    weights: dict[Action, float] = field(
        default_factory=lambda: {
            Action.CLASSIFY: 0.45,
            Action.CHEAP_VLM: 0.75,
            Action.STRONG_VLM: 1.35,
            Action.MULTIFRAME: 1.6,
        }
    )
    #: Belief above this counts as "hazard detected" for recall accounting.
    detect_threshold: float = 0.6


@dataclass(slots=True)
class SchedulerConfig:
    """The MAVIS decision rule itself."""

    #: Base value assigned to one bit of information, in units of cost.
    #: Score(a) = -cost(a) + eps_risk * E[IG(a) | memory].
    risk_value_base: float = 0.02
    #: Value of information scales up with current hazard risk.
    risk_value_slope: float = 0.12
    #: How many memories to recall per decision.
    memory_k: int = 5
    #: Blend factor between memory-derived IG and the static prior.
    #: 0 = ignore memory entirely, 1 = trust memory completely at full relevance.
    memory_trust: float = 0.8
    #: Prior expected information gain per action, used before memory warms up.
    prior_ig: dict[Action, float] = field(
        default_factory=lambda: {
            Action.SKIP: 0.0,
            Action.CLASSIFY: 0.05,
            Action.CHEAP_VLM: 0.14,
            Action.STRONG_VLM: 0.34,
            Action.MULTIFRAME: 0.42,
        }
    )
    #: Static cost prior per action, in the same arbitrary unit as measured cost.
    #: Overridden by observed costs once memory has them.
    prior_cost: dict[Action, float] = field(
        default_factory=lambda: {
            Action.SKIP: 0.0,
            Action.CLASSIFY: 0.0006,
            Action.CHEAP_VLM: 0.0018,
            Action.STRONG_VLM: 0.0090,
            Action.MULTIFRAME: 0.0155,
        }
    )

    # ---- recall safety floor -------------------------------------------
    # Pure score maximisation will happily starve recall. These constraints
    # bound how long MAVIS may go without paying for a strong look.

    #: Risk band in which the scene is genuinely ambiguous.
    ambiguous_band: tuple[float, float] = (0.30, 0.72)
    #: Inside the ambiguous band, force a strong call if none happened recently.
    ambiguous_max_gap_s: float = 4.0
    #: Absolute ceiling on time without a strong call, whatever the score says.
    #: Sized for continuous footage; see ``strong_max_gap_frac``.
    strong_max_gap_s: float = 12.0
    #: The absolute ceiling is useless on footage shorter than itself — a 12s
    #: ceiling on a 10s clip fires exactly once, which is how MAVIS ended up
    #: looking a single time per clip and losing every detection after the first
    #: frame. Scale the ceiling to the material as well: whichever of the two is
    #: tighter wins, floored so it can never collapse to zero.
    strong_max_gap_frac: float = 0.3
    strong_min_gap_s: float = 1.5
    #: Always spend a strong call on the first gated frame of a clip.
    strong_on_first_frame: bool = True


@dataclass(slots=True)
class BaselineConfig:
    """Fixed-rate strong-VLM baseline, the thing MAVIS has to beat on cost."""

    sample_every_s: float = 1.0


@dataclass(slots=True)
class Config:
    gate: GateConfig = field(default_factory=GateConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    #: JPEG quality used when handing a frame to Cortex.
    jpeg_quality: int = 80
    #: Longest edge of a frame sent for inference. Smaller = fewer prompt tokens.
    inference_max_edge: int = 768
    seed: int = 20260807


DEFAULT = Config()
