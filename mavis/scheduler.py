"""The MAVIS decision engine.

For each action the scheduler evaluates

    Score(a) = -C(a) + ε_risk · E[ΔH(a) | M_t]

where ``C(a)`` is what that action has actually cost in comparable situations,
``E[ΔH(a) | M_t]`` is the information gain that similar past episodes actually
delivered, and ``ε_risk`` is how many units of cost one bit of information is
worth right now — higher when the scene looks dangerous, because being wrong is
more expensive there.

Two properties matter more than the formula:

**Cold start degrades, it does not lie.** ``aggregate`` returns a support weight
alongside its estimate. With no memory, support is zero and the scheduler falls
back entirely to a static prior. Memory takes over gradually as support grows,
so an empty store yields a sane fixed-policy scheduler rather than a confident
one driven by a single episode.

**Recall has a floor that scoring cannot breach.** A pure argmax will happily
starve recall to save credits, and a benchmark that reports 82% cost reduction
against collapsed recall is worthless. :meth:`_safety_override` forces a strong
call when the scene is in the ambiguous risk band, when too long has passed
since the last strong look, and on the first gated frame of every clip. These
overrides are recorded on the decision (``forced=True``) so the benchmark can
report exactly how many of MAVIS's strong calls the policy chose versus how many
the floor demanded.

Decisions are made in two stages, which is where the saving actually comes from:

1. :meth:`decide_gate` — before spending anything, decide whether this frame is
   even worth a cheap AI_CLASSIFY. SKIP here costs literally zero.
2. :meth:`decide_escalation` — with the cheap scene summary and the memories it
   retrieved, decide whether to pay for real reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .belief import HazardBelief
from .config import SchedulerConfig
from .memory.base import MemoryStore, aggregate, scene_similarity
from .types import Action, Decision, MemoryHit, SceneSummary

#: Actions that :meth:`decide_escalation` chooses between.
ESCALATIONS = [Action.SKIP, Action.CHEAP_VLM, Action.STRONG_VLM, Action.MULTIFRAME]


@dataclass(slots=True)
class DecisionContext:
    """Everything about *when* this decision is happening."""

    timestamp_s: float
    is_first_gated_frame: bool = False
    time_since_strong_s: float = 1e9
    time_since_classify_s: float = 1e9
    have_previous_frame: bool = False
    prev_scene: SceneSummary | None = None
    #: Length of the footage being processed, used to size the recall floor.
    #: 0 means unknown, in which case only the absolute ceiling applies.
    clip_duration_s: float = 0.0


@dataclass(slots=True)
class Estimate:
    ig: float
    cost: float
    support: float
    from_memory: bool = False


@dataclass(slots=True)
class MavisScheduler:
    """Memory-conditioned scheduler. Stateless across clips except via memory."""

    cfg: SchedulerConfig
    memory: MemoryStore
    last_hits: list[MemoryHit] = field(default_factory=list)

    # -- stage 1: is this frame worth even a cheap look? -------------------

    def decide_gate(self, belief: HazardBelief, ctx: DecisionContext) -> Decision:
        """Choose between SKIP and CLASSIFY, before any spend.

        With no fresh scene yet, the only evidence is the previous scene, the
        current belief and elapsed time. Memory is queried against the previous
        scene — "last time the camera looked like this, was another cheap look
        worth it?"
        """
        if ctx.is_first_gated_frame and self.cfg.strong_on_first_frame:
            return Decision(
                action=Action.CLASSIFY,
                reason="first gated frame: establish a scene baseline",
                forced=True,
            )

        # The recall floor has to be reachable from here. Stage 2 is where strong
        # calls are forced, but stage 2 only runs if stage 1 spent something — so
        # a stage 1 that keeps skipping would silently starve recall and the floor
        # would never fire. When a strong look is overdue, buy the cheap look that
        # leads to one.
        overdue = self._strong_overdue(belief.p, ctx)
        if overdue:
            return Decision(action=Action.CLASSIFY, reason=overdue, forced=True)

        hits = self.memory.search(ctx.prev_scene, self.cfg.memory_k) if ctx.prev_scene else []
        self.last_hits = hits

        risk_signal = max(belief.p, ctx.prev_scene.risk if ctx.prev_scene else 0.0)
        eps = self._epsilon(risk_signal)

        est = self._estimate(Action.CLASSIFY, hits, belief)
        # How much of what we learned last time has gone stale? Use the belief's
        # own decay curve rather than an independent guess, so stage 1 and the
        # belief model cannot disagree: looking again before anything has decayed
        # genuinely cannot tell us something new.
        decay = belief.cfg.decay_per_s
        staleness = 1.0 - math.exp(-decay * max(ctx.time_since_classify_s, 0.0))
        ig = est.ig * staleness

        score = -est.cost + eps * ig
        scores = {Action.SKIP.value: 0.0, Action.CLASSIFY.value: score}

        if score <= 0:
            return Decision(
                action=Action.SKIP,
                reason=(
                    f"redundant scene: expected gain {ig:.3f} bits does not cover "
                    f"{est.cost:.5f} at ε={eps:.3f}"
                ),
                scores=scores,
                expected_ig=ig,
                expected_cost=est.cost,
                memory_hits=len(hits),
            )
        return Decision(
            action=Action.CLASSIFY,
            reason=f"cheap look worthwhile: expected {ig:.3f} bits (memory n={len(hits)})",
            scores=scores,
            expected_ig=ig,
            expected_cost=est.cost,
            memory_hits=len(hits),
        )

    # -- stage 2: is real reasoning worth paying for? ----------------------

    def decide_escalation(
        self, scene: SceneSummary, belief: HazardBelief, ctx: DecisionContext
    ) -> Decision:
        """Choose among SKIP / CHEAP_VLM / STRONG_VLM / MULTIFRAME."""
        hits = self.memory.search(scene, self.cfg.memory_k)
        self.last_hits = hits

        forced = self._safety_override(scene, belief, ctx)
        risk_signal = max(belief.p, scene.risk)
        eps = self._epsilon(risk_signal)
        novelty = self._novelty(scene, ctx)

        scores: dict[str, float] = {}
        estimates: dict[Action, Estimate] = {}
        for action in ESCALATIONS:
            if action is Action.MULTIFRAME and not ctx.have_previous_frame:
                continue
            if action is Action.SKIP:
                scores[action.value] = 0.0
                estimates[action] = Estimate(0.0, 0.0, 0.0)
                continue
            est = self._estimate(action, hits, belief)
            ig = est.ig * novelty
            scores[action.value] = -est.cost + eps * ig
            estimates[action] = Estimate(ig, est.cost, est.support, est.from_memory)

        best = max(scores, key=lambda k: scores[k])
        chosen = Action(best)

        if forced and chosen not in (Action.STRONG_VLM, Action.MULTIFRAME):
            est = estimates.get(Action.STRONG_VLM, Estimate(0.0, 0.0, 0.0))
            return Decision(
                action=Action.STRONG_VLM,
                reason=forced,
                scores=scores,
                expected_ig=est.ig,
                expected_cost=est.cost,
                memory_hits=len(hits),
                forced=True,
            )

        est = estimates[chosen]
        support = est.support
        source = f"memory n={len(hits)} support={support:.2f}" if support > 0 else "prior (cold)"
        reason = (
            f"skip reasoning: no action clears its cost at ε={eps:.3f}"
            if chosen is Action.SKIP
            else f"expected {est.ig:.3f} bits for {est.cost:.5f} at ε={eps:.3f} [{source}]"
        )
        return Decision(
            action=chosen,
            reason=reason,
            scores=scores,
            expected_ig=est.ig,
            expected_cost=est.cost,
            memory_hits=len(hits),
        )

    # -- internals ---------------------------------------------------------

    def _epsilon(self, risk_signal: float) -> float:
        """How much one bit of information is worth, in cost units, right now."""
        return self.cfg.risk_value_base + self.cfg.risk_value_slope * risk_signal

    def _estimate(self, action: Action, hits: list[MemoryHit], belief: HazardBelief) -> Estimate:
        """Blend what memory measured with the static prior, weighted by support."""
        ig_mem, cost_mem, support = aggregate(hits, action)
        trust = self.cfg.memory_trust * (support / (support + 1.0))

        ig = trust * ig_mem + (1.0 - trust) * self.cfg.prior_ig[action]
        cost = (
            trust * cost_mem + (1.0 - trust) * self.cfg.prior_cost[action]
            if cost_mem > 0
            else self.cfg.prior_cost[action]
        )
        # An observation can only reveal what is still uncertain, and it reveals
        # proportionally less as the belief settles. Scaling by the remaining
        # entropy (at most 1 bit for a binary belief) rather than merely clipping
        # to it is what stops a confident scene from still looking worth paying
        # for: with H=0.14 bits left, a nominal 0.14-bit action is worth 0.02.
        ig = min(ig * belief.entropy, belief.entropy)
        return Estimate(ig=ig, cost=cost, support=support, from_memory=support > 0)

    def _novelty(self, scene: SceneSummary, ctx: DecisionContext) -> float:
        """Discount expected gain when this frame restates the previous one."""
        if ctx.prev_scene is None:
            return 1.0
        return 0.4 + 0.6 * (1.0 - scene_similarity(scene, ctx.prev_scene))

    def _strong_overdue(self, risk: float, ctx: DecisionContext) -> str | None:
        """Is a strong look mandatory on time-since grounds alone?

        Shared by both stages so the floor cannot be bypassed by skipping early.
        """
        ceiling = self._ceiling(ctx)
        low, high = self.cfg.ambiguous_band
        ambiguous_gap = min(self.cfg.ambiguous_max_gap_s, ceiling)
        if low <= risk <= high and ctx.time_since_strong_s >= ambiguous_gap:
            return (
                f"recall floor: risk {risk:.2f} inside ambiguous band "
                f"[{low:.2f}, {high:.2f}] and {ctx.time_since_strong_s:.1f}s since a strong look"
            )
        if ctx.time_since_strong_s >= ceiling:
            return (
                f"recall floor: {ctx.time_since_strong_s:.1f}s since a strong look "
                f"(ceiling {ceiling:.1f}s)"
            )
        return None

    def _ceiling(self, ctx: DecisionContext) -> float:
        """Longest permissible gap between strong looks, for this footage."""
        ceiling = self.cfg.strong_max_gap_s
        if ctx.clip_duration_s > 0:
            scaled = ctx.clip_duration_s * self.cfg.strong_max_gap_frac
            ceiling = min(ceiling, max(self.cfg.strong_min_gap_s, scaled))
        return ceiling

    def _safety_override(
        self, scene: SceneSummary, belief: HazardBelief, ctx: DecisionContext
    ) -> str | None:
        """The recall floor. Returns a reason string when a strong call is mandatory."""
        if ctx.is_first_gated_frame and self.cfg.strong_on_first_frame:
            return "recall floor: first gated frame of the clip"
        # The cheap tier has now spoken, so the ambiguity test uses its risk too.
        return self._strong_overdue(max(belief.p, scene.risk), ctx)
