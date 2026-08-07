"""Policy execution: run one clip through one policy and record everything.

Both policies share the same decoder, the same Cortex client, the same prompt and
the same belief update. The only difference between them is which frames get an
expensive call — which is precisely the claim under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .belief import HazardBelief
from .config import Config
from .cortex.base import HAZARD_PROMPT, MULTIFRAME_PROMPT
from .dataset import CLASSIFY_CATEGORIES, Clip
from .gate import GateStats, iter_gated_frames, probe
from .memory.base import MemoryStore
from .scheduler import DecisionContext, MavisScheduler
from .types import (
    Action,
    CostRecord,
    Decision,
    Episode,
    Frame,
    InferenceResult,
    SceneSummary,
    Step,
    Trace,
)

StepCallback = Callable[[Step, np.ndarray], None]


@dataclass(slots=True)
class _ClipState:
    belief: HazardBelief
    steps: list[Step]
    prev_scene: SceneSummary | None = None
    prev_image: np.ndarray | None = None
    last_strong_ts: float = -1e9
    last_classify_ts: float = -1e9


def _reason(cortex, action: Action, images, prompt: str):
    """Dispatch one action to Cortex. Returns ``(result, cost)``."""
    if action is Action.CLASSIFY:
        return cortex.classify(images[0], CLASSIFY_CATEGORIES)
    strong = action in (Action.STRONG_VLM, Action.MULTIFRAME)
    return cortex.complete(images, prompt, strong=strong)


def run_baseline(
    clip: Clip,
    cortex,
    cfg: Config,
    *,
    use_gate: bool = False,
    on_step: StepCallback | None = None,
) -> Trace:
    """Fixed-rate sampling, strong VLM on every sampled frame.

    ``use_gate=False`` is the baseline as normally described — sample the video
    at a fixed rate and reason about every sample. ``use_gate=True`` gives the
    baseline the same free OpenCV filter MAVIS gets, which is the harder and
    fairer comparison; the benchmark reports both so nobody has to take the
    easier one on trust.
    """
    duration_s, _fps, _count = probe(clip.path)
    if hasattr(cortex, "set_scene_truth"):
        cortex.set_scene_truth(clip.class_id)

    state = _ClipState(belief=HazardBelief(cfg.belief), steps=[])
    gate_stats = GateStats()
    next_sample_ts = 0.0

    for frame, stats in _frames(clip, cfg, gated=use_gate):
        gate_stats = stats
        if frame.timestamp_s + 1e-9 < next_sample_ts:
            continue
        next_sample_ts = frame.timestamp_s + cfg.baseline.sample_every_s

        decision = Decision(
            action=Action.STRONG_VLM,
            reason=f"baseline: fixed {cfg.baseline.sample_every_s:.1f}s sampling",
        )
        step = _execute(cortex, state, frame, decision, HAZARD_PROMPT)
        state.steps.append(step)
        if on_step:
            on_step(step, frame.image)

    return Trace(
        clip_id=clip.clip_id,
        policy="baseline",
        label_hazard=clip.is_hazard,
        label_class=clip.class_folder,
        duration_s=duration_s,
        frames_decoded=gate_stats.decoded,
        frames_gated=gate_stats.passed,
        steps=state.steps,
    )


def run_mavis(
    clip: Clip,
    cortex,
    memory: MemoryStore,
    cfg: Config,
    *,
    learn: bool = True,
    outcome_source: str = "belief",
    on_step: StepCallback | None = None,
) -> Trace:
    """Memory-conditioned adaptive scheduling.

    ``outcome_source`` controls what gets written back as the hazard outcome of
    each episode. ``"belief"`` uses the system's own final verdict, which is the
    only option at inference time. ``"label"`` uses the clip's ground-truth
    class and is valid **only** when warming memory on the train split — using it
    on the evaluation split would leak labels into the scheduler.
    """
    if outcome_source not in ("belief", "label"):
        raise ValueError(f"outcome_source must be 'belief' or 'label', got {outcome_source!r}")

    duration_s, _fps, _count = probe(clip.path)
    if hasattr(cortex, "set_scene_truth"):
        cortex.set_scene_truth(clip.class_id)

    scheduler = MavisScheduler(cfg.scheduler, memory)
    state = _ClipState(belief=HazardBelief(cfg.belief), steps=[])
    gate_stats = GateStats()
    pending: list[Episode] = []
    first = True

    for frame, stats in _frames(clip, cfg, gated=True):
        gate_stats = stats
        ctx = DecisionContext(
            timestamp_s=frame.timestamp_s,
            is_first_gated_frame=first,
            time_since_strong_s=frame.timestamp_s - state.last_strong_ts,
            time_since_classify_s=frame.timestamp_s - state.last_classify_ts,
            have_previous_frame=state.prev_image is not None,
            prev_scene=state.prev_scene,
            clip_duration_s=duration_s,
        )
        first = False

        # Stage 1 — is this frame worth even a cheap look?
        gate_decision = scheduler.decide_gate(state.belief, ctx)
        if gate_decision.action is Action.SKIP:
            state.belief.decay_to(frame.timestamp_s)
            step = Step(
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                decision=gate_decision,
                hazard_prob_before=state.belief.p,
                hazard_prob_after=state.belief.p,
            )
            state.steps.append(step)
            if on_step:
                on_step(step, frame.image)
            continue

        classify_step = _execute(cortex, state, frame, gate_decision, HAZARD_PROMPT)
        state.last_classify_ts = frame.timestamp_s
        state.prev_scene = classify_step.scene
        state.steps.append(classify_step)
        if on_step:
            on_step(classify_step, frame.image)
        if learn and classify_step.scene:
            pending.append(_episode(clip, classify_step, classify_step.scene))

        # Stage 2 — with the cheap summary in hand, is real reasoning worth it?
        scene = classify_step.scene or SceneSummary()
        ctx.prev_scene = state.prev_scene
        ctx.time_since_strong_s = frame.timestamp_s - state.last_strong_ts
        escalation = scheduler.decide_escalation(scene, state.belief, ctx)
        if escalation.action is Action.SKIP:
            # Still the most recent frame we actually looked at, so MULTIFRAME
            # later compares against this one rather than a much older frame.
            state.prev_image = frame.image
            continue

        images = [frame.image]
        prompt = HAZARD_PROMPT
        if escalation.action is Action.MULTIFRAME and state.prev_image is not None:
            images = [state.prev_image, frame.image]
            prompt = MULTIFRAME_PROMPT

        step = _execute(cortex, state, frame, escalation, prompt, images=images)
        if escalation.action in (Action.STRONG_VLM, Action.MULTIFRAME):
            state.last_strong_ts = frame.timestamp_s
        state.steps.append(step)
        if on_step:
            on_step(step, frame.image)
        if learn:
            pending.append(_episode(clip, step, scene))

        state.prev_image = frame.image

    if learn and pending:
        outcome = clip.is_hazard if outcome_source == "label" else state.belief.peak >= cfg.belief.detect_threshold
        for episode in pending:
            episode.hazard_outcome = outcome
            memory.write(episode)
        memory.flush()

    return Trace(
        clip_id=clip.clip_id,
        policy="mavis",
        label_hazard=clip.is_hazard,
        label_class=clip.class_folder,
        duration_s=duration_s,
        frames_decoded=gate_stats.decoded,
        frames_gated=gate_stats.passed,
        steps=state.steps,
    )


# -- shared machinery ------------------------------------------------------


def _frames(clip: Clip, cfg: Config, *, gated: bool) -> Iterator[tuple[Frame, GateStats]]:
    if gated:
        yield from iter_gated_frames(clip.path, cfg.gate)
        return
    # Ungated: still stride-decode, but let everything through.
    import cv2

    cap = cv2.VideoCapture(str(clip.path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {clip.path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stats = GateStats()
    index = 0
    try:
        while True:
            if not cap.grab():
                break
            if index % cfg.gate.decode_stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                stats.decoded += 1
                stats.passed += 1
                yield Frame(index=index, timestamp_s=index / fps, image=image), stats
            index += 1
    finally:
        cap.release()


def _execute(
    cortex,
    state: _ClipState,
    frame: Frame,
    decision: Decision,
    prompt: str,
    images: list[np.ndarray] | None = None,
) -> Step:
    """Run one paid action, fold the answer into the belief, measure the gain."""
    images = images or [frame.image]
    before = state.belief.p

    result_or_scene, cost = _reason(cortex, decision.action, images, prompt)

    scene: SceneSummary | None = None
    inference: InferenceResult | None = None
    observed_ig = 0.0

    if decision.action is Action.CLASSIFY:
        scene = result_or_scene
        # AI_CLASSIFY answers "what is this", not "is it dangerous". Treat its
        # risk score as a weak observation so the belief still moves a little.
        inference = InferenceResult(
            hazard_prob=scene.risk, confidence=0.35, rationale="cheap classification"
        )
        observed_ig = state.belief.observe(inference, Action.CLASSIFY, frame.timestamp_s)
    else:
        inference = result_or_scene
        observed_ig = state.belief.observe(inference, decision.action, frame.timestamp_s)

    return Step(
        frame_index=frame.index,
        timestamp_s=frame.timestamp_s,
        decision=decision,
        scene=scene,
        inference=inference,
        cost=cost,
        hazard_prob_before=before,
        hazard_prob_after=state.belief.p,
        observed_ig=observed_ig,
    )


def _episode(clip: Clip, step: Step, scene: SceneSummary) -> Episode:
    """Package one executed step for memory. Outcome is filled in at clip end."""
    return Episode(
        scene=scene,
        action=step.decision.action,
        observed_ig=step.observed_ig,
        observed_cost=_cost_scalar(step.cost),
        hazard_outcome=False,  # overwritten once the clip resolves
        hazard_prob_before=step.hazard_prob_before,
        hazard_prob_after=step.hazard_prob_after,
        clip_id=clip.clip_id,
        timestamp_s=step.timestamp_s,
    )


def _cost_scalar(cost: CostRecord | None) -> float:
    """Reduce a cost record to the single number the scheduler trades against.

    Credits when Snowflake has reported them, otherwise tokens scaled into the
    same range so a run is still self-consistent before usage history lands.
    """
    if cost is None:
        return 0.0
    if cost.credits is not None:
        return cost.credits
    return cost.total_tokens / 1e6
