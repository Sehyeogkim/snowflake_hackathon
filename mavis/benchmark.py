"""Baseline vs MAVIS on the same clips, under the same conditions.

The experiment has three phases:

1. **Warm-up** (optional, train split only). MAVIS runs over training clips and
   writes episodes to memory. Ground-truth labels are used as the episode outcome
   here and only here — it is a labelled training set, and saying so plainly is
   better than pretending memory bootstraps itself from nothing.
2. **Baseline** over the evaluation split.
3. **MAVIS** over the same evaluation split, with learning disabled so the
   measured policy is fixed for the whole evaluation.

Holding the split, the clip order, the Cortex client, the prompt and the belief
model constant across (2) and (3) is what makes the comparison mean anything.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from . import calibrate as calibrate_mod, dataset
from .config import Config
from .dataset import Clip

log = logging.getLogger(__name__)
from .memory.base import MemoryStore
from .metrics import Comparison, PolicyMetrics, evaluate
from .runner import run_baseline, run_mavis
from .types import Trace


@dataclass(slots=True)
class BenchmarkResult:
    comparison: Comparison
    baseline_traces: list[Trace]
    mavis_traces: list[Trace]
    warmup_clips: int
    memory_size: int
    #: Cost of measuring the action priors. Reported apart from both policies:
    #: neither of them made these calls, so folding it into either would misstate
    #: what that policy actually spent.
    calibration_cost: float = 0.0
    calibrated_costs: dict[str, float] | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "warmup_clips": self.warmup_clips,
            "memory_size": self.memory_size,
            "calibration_cost": self.calibration_cost,
            "calibrated_costs": self.calibrated_costs,
            "estimated": self.comparison.estimated,
            "backend": self.comparison.backend,
            "cost_unit": self.comparison.cost_unit,
            "cost_basis": self.comparison.cost_basis,
            "cost_reduction": self.comparison.cost_reduction,
            "recall_delta_pp": self.comparison.recall_delta_pp,
            "baseline": _metrics_dict(self.comparison.baseline),
            "mavis": _metrics_dict(self.comparison.mavis),
            "traces": {
                "baseline": [t.to_dict() for t in self.baseline_traces],
                "mavis": [t.to_dict() for t in self.mavis_traces],
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def warm_up(clips: list[Clip], cortex, memory: MemoryStore, cfg: Config) -> int:
    """Populate memory from labelled training clips. Returns clips processed."""
    clips, unreadable = dataset.readable(clips)
    if unreadable:
        log.warning("warm-up skipping %d unreadable clip(s)", len(unreadable))
    for clip in clips:
        run_mavis(clip, cortex, memory, cfg, learn=True, outcome_source="label")
    return len(clips)


def run(
    eval_clips: list[Clip],
    cortex,
    memory: MemoryStore,
    cfg: Config,
    *,
    warmup_clips: list[Clip] | None = None,
    baseline_gated: bool = False,
    shuffle_seed: int | None = None,
) -> BenchmarkResult:
    """Execute the full experiment and return both policies' traces."""
    if not eval_clips:
        raise ValueError("no evaluation clips found — has the dataset finished downloading?")

    eval_clips, unreadable = dataset.readable(eval_clips)
    if unreadable:
        # Loud, not silent: a truncated download must not quietly shrink the
        # evaluation set and make the result look better than it was measured on.
        log.warning(
            "skipping %d unreadable clip(s) (likely truncated downloads): %s",
            len(unreadable),
            ", ".join(c.path.name for c in unreadable),
        )
    if not eval_clips:
        raise ValueError("every evaluation clip is unreadable — re-run the fetch script")

    if shuffle_seed is not None:
        eval_clips = list(eval_clips)
        random.Random(shuffle_seed).shuffle(eval_clips)

    # Measure what each action costs on this backend before scheduling anything
    # against invented numbers. Uses a real frame: image tokens dominate the bill
    # and scale with resolution.
    calibration = None
    frame = calibrate_mod.sample_frame(eval_clips, cfg)
    if frame is not None:
        calibration = calibrate_mod.calibrate(cortex, cfg, frame)
        if calibration:
            log.info("%s", calibration.describe())

    warmed = warm_up(warmup_clips, cortex, memory, cfg) if warmup_clips else 0

    baseline_traces = [
        run_baseline(clip, cortex, cfg, use_gate=baseline_gated) for clip in eval_clips
    ]
    # learn=False: the policy under measurement must not change mid-evaluation.
    mavis_traces = [run_mavis(clip, cortex, memory, cfg, learn=False) for clip in eval_clips]

    threshold = cfg.belief.detect_threshold
    b = evaluate(baseline_traces, threshold)
    m = evaluate(mavis_traces, threshold)

    return BenchmarkResult(
        comparison=Comparison(
            baseline=b,
            mavis=m,
            estimated=getattr(cortex, "estimated", True),
            cost_unit=getattr(cortex, "cost_unit", "credits"),
            backend=getattr(cortex, "name", "unknown"),
        ),
        baseline_traces=baseline_traces,
        mavis_traces=mavis_traces,
        warmup_clips=warmed,
        memory_size=len(memory),
        calibration_cost=calibration.total_cost if calibration else 0.0,
        calibrated_costs=(
            {a.value: c for a, c in calibration.costs.items()} if calibration else None
        ),
    )


def _metrics_dict(m: PolicyMetrics) -> dict:
    return {
        "policy": m.policy,
        "clips": m.clips,
        "hazard_clips": m.hazard_clips,
        "safe_clips": m.safe_clips,
        "recall": m.recall,
        "false_positive_rate": m.false_positive_rate,
        "mean_latency_s": m.mean_latency,
        "cortex_calls": m.cortex_calls,
        "strong_calls": m.strong_calls,
        "cheap_calls": m.cheap_calls,
        "calls_by_action": m.calls,
        "prompt_tokens": m.prompt_tokens,
        "completion_tokens": m.completion_tokens,
        "total_tokens": m.total_tokens,
        "credits": m.credits,
        "credits_measured": m.credits_measured,
        "frames_decoded": m.frames_decoded,
        "frames_gated": m.frames_gated,
        "forced_strong": m.forced_strong,
        "chosen_strong": m.chosen_strong,
    }
