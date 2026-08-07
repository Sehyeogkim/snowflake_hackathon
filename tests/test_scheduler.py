"""The scheduler is the claim. These tests cover the three behaviours that make
the claim either true or hollow: it must skip when memory says an observation
was worthless, it must still work with no memory at all, and no amount of
score-chasing may be allowed to breach the recall floor.
"""

from __future__ import annotations

from mavis.belief import HazardBelief
from mavis.config import BeliefConfig, SchedulerConfig
from mavis.memory.local import LocalMemory
from mavis.scheduler import DecisionContext, MavisScheduler
from mavis.types import Action, Episode, MemoryHit, SceneSummary


def scene(risk: float = 0.4, labels=("forklift handling a load",)) -> SceneSummary:
    return SceneSummary(labels=list(labels), objects=["forklift", "pallet"], risk=risk)


def empty_memory() -> LocalMemory:
    return LocalMemory(path=None, load=False)


def scheduler(memory=None, cfg: SchedulerConfig | None = None) -> MavisScheduler:
    return MavisScheduler(cfg or SchedulerConfig(), memory or empty_memory())


def belief(p: float | None = None) -> HazardBelief:
    cfg = BeliefConfig()
    b = HazardBelief(cfg)
    if p is not None:
        b._logit = __import__("math").log(p / (1 - p))  # noqa: SLF001 - test fixture
    return b


class FakeMemory:
    """Memory that returns a fixed set of hits, to control the scheduler's prior."""

    name = "fake"

    def __init__(self, hits: list[MemoryHit]):
        self.hits = hits

    def search(self, scene, k):
        return self.hits[:k]

    def write(self, episode):
        pass

    def flush(self):
        pass

    def __len__(self):
        return len(self.hits)


# -- cold start ------------------------------------------------------------


def test_cold_start_still_produces_a_decision():
    sched = scheduler()
    d = sched.decide_escalation(scene(), belief(), DecisionContext(timestamp_s=5.0))
    assert d.action in (Action.SKIP, Action.CHEAP_VLM, Action.STRONG_VLM)
    assert d.memory_hits == 0


def test_high_risk_scene_buys_a_strong_look():
    sched = scheduler()
    ctx = DecisionContext(timestamp_s=5.0, time_since_strong_s=0.5)
    d = sched.decide_escalation(scene(risk=0.9), belief(0.55), ctx)
    assert d.action in (Action.STRONG_VLM, Action.MULTIFRAME)


def test_low_risk_settled_scene_does_not_buy_reasoning():
    # Belief already confident that nothing is wrong, cheap tier agrees, and a
    # strong call happened recently: paying again buys almost nothing.
    sched = scheduler()
    ctx = DecisionContext(timestamp_s=5.0, time_since_strong_s=0.5)
    d = sched.decide_escalation(scene(risk=0.03), belief(0.02), ctx)
    assert d.action is Action.SKIP


# -- memory changes the decision -------------------------------------------


def test_memory_of_worthless_calls_suppresses_them():
    """The whole point: if strong calls on this kind of scene never paid off,
    stop buying them."""
    hits = [
        MemoryHit(
            action=Action.STRONG_VLM,
            observed_ig=0.001,
            observed_cost=0.02,
            hazard_outcome=False,
            relevance=0.95,
            confidence=0.9,
        )
        for _ in range(5)
    ]
    warm = scheduler(FakeMemory(hits))
    cold = scheduler()
    ctx = DecisionContext(timestamp_s=5.0, time_since_strong_s=0.5)

    warm_d = warm.decide_escalation(scene(risk=0.5), belief(0.35), ctx)
    cold_d = cold.decide_escalation(scene(risk=0.5), belief(0.35), ctx)

    assert warm_d.scores[Action.STRONG_VLM.value] < cold_d.scores[Action.STRONG_VLM.value]
    assert warm_d.expected_ig < cold_d.expected_ig


def test_memory_of_valuable_calls_encourages_them():
    hits = [
        MemoryHit(
            action=Action.STRONG_VLM,
            observed_ig=0.55,
            observed_cost=0.002,
            hazard_outcome=True,
            relevance=0.95,
            confidence=0.9,
        )
        for _ in range(5)
    ]
    warm = scheduler(FakeMemory(hits))
    ctx = DecisionContext(timestamp_s=5.0, time_since_strong_s=0.5)
    d = warm.decide_escalation(scene(risk=0.45), belief(0.3), ctx)
    assert d.action in (Action.STRONG_VLM, Action.MULTIFRAME)
    assert d.memory_hits == 5


# -- the recall floor ------------------------------------------------------


def test_ambiguous_scene_forces_a_strong_call_after_the_gap():
    """Memory saying 'never worth it' must not be able to starve recall."""
    hits = [
        MemoryHit(
            action=a,
            observed_ig=0.0,
            observed_cost=0.05,
            hazard_outcome=False,
            relevance=1.0,
            confidence=1.0,
        )
        for a in (Action.STRONG_VLM, Action.CHEAP_VLM, Action.MULTIFRAME)
    ]
    sched = scheduler(FakeMemory(hits))
    cfg = sched.cfg
    ctx = DecisionContext(
        timestamp_s=20.0, time_since_strong_s=cfg.ambiguous_max_gap_s + 0.1
    )
    d = sched.decide_escalation(scene(risk=0.5), belief(0.45), ctx)
    assert d.action is Action.STRONG_VLM
    assert d.forced
    assert "recall floor" in d.reason


def test_absolute_ceiling_forces_a_strong_call_even_when_risk_looks_low():
    sched = scheduler()
    cfg = sched.cfg
    ctx = DecisionContext(
        timestamp_s=60.0, time_since_strong_s=cfg.strong_max_gap_s + 1.0
    )
    d = sched.decide_escalation(scene(risk=0.01), belief(0.01), ctx)
    assert d.action is Action.STRONG_VLM
    assert d.forced


def test_first_gated_frame_always_classifies():
    sched = scheduler()
    d = sched.decide_gate(belief(), DecisionContext(timestamp_s=0.0, is_first_gated_frame=True))
    assert d.action is Action.CLASSIFY
    assert d.forced


# -- stage 1 -----------------------------------------------------------------


def test_gate_skips_a_scene_it_just_looked_at():
    sched = scheduler()
    prev = scene(risk=0.05)
    ctx = DecisionContext(
        timestamp_s=1.0,
        prev_scene=prev,
        time_since_classify_s=0.05,
        time_since_strong_s=0.5,  # otherwise the recall floor fires, correctly
    )
    d = sched.decide_gate(belief(0.03), ctx)
    assert d.action is Action.SKIP
    assert d.expected_cost == 0.0 or d.scores[Action.SKIP.value] == 0.0


def test_gate_looks_again_once_the_scene_is_stale():
    sched = scheduler()
    prev = scene(risk=0.5)
    ctx = DecisionContext(
        timestamp_s=30.0,
        prev_scene=prev,
        time_since_classify_s=25.0,
        time_since_strong_s=0.5,
    )
    d = sched.decide_gate(belief(0.4), ctx)
    assert d.action is Action.CLASSIFY


def test_the_recall_floor_reaches_through_stage_one():
    """Stage 2 is where strong calls are forced, but stage 2 only runs if stage 1
    spent something. A stage 1 that keeps skipping would starve recall silently,
    so an overdue strong look must force a cheap look first."""
    sched = scheduler()
    prev = scene(risk=0.02)
    ctx = DecisionContext(
        timestamp_s=60.0,
        prev_scene=prev,
        time_since_classify_s=0.05,  # nothing has gone stale
        time_since_strong_s=sched.cfg.strong_max_gap_s + 1.0,  # but a look is overdue
    )
    d = sched.decide_gate(belief(0.02), ctx)
    assert d.action is Action.CLASSIFY
    assert d.forced
    assert "recall floor" in d.reason


# -- memory round trip -------------------------------------------------------


def test_local_memory_recalls_what_it_wrote(tmp_path):
    store = LocalMemory(path=tmp_path / "episodes.jsonl")
    s = scene(risk=0.6)
    store.write(
        Episode(
            scene=s,
            action=Action.STRONG_VLM,
            observed_ig=0.4,
            observed_cost=0.003,
            hazard_outcome=True,
            hazard_prob_before=0.2,
            hazard_prob_after=0.8,
            clip_id="test/clip",
        )
    )
    hits = store.search(s, k=5)
    assert len(hits) == 1
    assert hits[0].action is Action.STRONG_VLM
    assert hits[0].relevance > 0.9

    reloaded = LocalMemory(path=tmp_path / "episodes.jsonl")
    assert len(reloaded) == 1


def test_unrelated_scene_is_not_recalled(tmp_path):
    store = LocalMemory(path=tmp_path / "episodes.jsonl")
    store.write(
        Episode(
            scene=SceneSummary(labels=["electrical panel in view"], objects=["panel"], risk=0.7),
            action=Action.STRONG_VLM,
            observed_ig=0.4,
            observed_cost=0.003,
            hazard_outcome=True,
            hazard_prob_before=0.2,
            hazard_prob_after=0.8,
        )
    )
    hits = store.search(SceneSummary(labels=["forklift handling a load"], objects=["forklift"], risk=0.1), k=5)
    assert all(h.relevance < 0.4 for h in hits)
