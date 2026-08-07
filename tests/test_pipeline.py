"""End-to-end tests over synthetic video.

These do not need the Mendeley download: they generate clips with OpenCV, which
also lets them assert things a real clip cannot guarantee — an entirely static
video must produce almost no gated frames, and MAVIS must never spend more than
the baseline on the same input.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mavis import benchmark
from mavis.config import Config
from mavis.cortex.mock import MockCortex
from mavis.dataset import Clip
from mavis.gate import iter_gated_frames, probe
from mavis.memory.local import LocalMemory
from mavis.metrics import evaluate
from mavis.runner import run_baseline, run_mavis
from mavis.types import Action

FPS = 25
SIZE = (240, 320)


def write_video(path, frames):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE[1], SIZE[0])
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def static_frames(n=150):
    base = np.full((*SIZE, 3), 120, dtype=np.uint8)
    cv2.rectangle(base, (40, 40), (120, 160), (60, 60, 200), -1)
    return [base.copy() for _ in range(n)]


def moving_frames(n=150):
    out = []
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = np.full((*SIZE, 3), 120, dtype=np.uint8)
        x = 10 + (i * 2) % (SIZE[1] - 70)
        cv2.rectangle(frame, (x, 60), (x + 60, 140), (60, 60, 200), -1)
        frame = np.clip(frame.astype(np.int16) + rng.integers(-8, 8, frame.shape), 0, 255)
        out.append(frame.astype(np.uint8))
    return out


@pytest.fixture
def hazard_clip(tmp_path):
    path = tmp_path / "hazard.mp4"
    write_video(path, moving_frames())
    return Clip(path=path, class_id=3, split="test")  # forklift overload = hazard


@pytest.fixture
def safe_clip(tmp_path):
    path = tmp_path / "safe.mp4"
    write_video(path, moving_frames())
    return Clip(path=path, class_id=7, split="test")  # safe carrying


@pytest.fixture
def static_clip(tmp_path):
    path = tmp_path / "static.mp4"
    write_video(path, static_frames())
    return Clip(path=path, class_id=7, split="test")


@pytest.fixture
def long_hazard_clip(tmp_path):
    """20 seconds — long enough to contain the redundancy MAVIS exploits."""
    path = tmp_path / "long_hazard.mp4"
    write_video(path, moving_frames(500))
    return Clip(path=path, class_id=3, split="test")


# -- gate --------------------------------------------------------------------


def test_probe_reports_a_sane_duration(hazard_clip):
    duration, fps, count = probe(hazard_clip.path)
    assert count > 0
    assert 4.0 < duration < 8.0


def test_gate_drops_almost_everything_in_a_static_clip(static_clip):
    cfg = Config()
    passed = list(iter_gated_frames(static_clip.path, cfg.gate))
    _frame, stats = passed[-1]
    # Only the first frame plus the periodic max-gap wakeups should survive.
    assert stats.passed <= stats.forced_by_gap + 2
    assert stats.drop_rate > 0.7


def test_gate_keeps_frames_in_a_moving_clip(hazard_clip):
    cfg = Config()
    passed = list(iter_gated_frames(hazard_clip.path, cfg.gate))
    assert len(passed) > 3


# -- runners -----------------------------------------------------------------


def test_baseline_calls_strong_on_every_sample(hazard_clip):
    cfg = Config()
    trace = run_baseline(hazard_clip, MockCortex(seed=1), cfg)
    assert trace.steps
    assert all(s.decision.action is Action.STRONG_VLM for s in trace.steps)
    assert trace.total_tokens > 0


def test_mavis_runs_and_records_every_decision(hazard_clip):
    cfg = Config()
    trace = run_mavis(hazard_clip, MockCortex(seed=1), LocalMemory(path=None, load=False), cfg)
    assert trace.steps
    actions = {s.decision.action for s in trace.steps}
    assert actions & {Action.CLASSIFY, Action.STRONG_VLM, Action.SKIP}
    for step in trace.steps:
        assert step.decision.reason  # every decision must be explainable


def test_mavis_costs_less_than_the_baseline_on_a_normal_clip(long_hazard_clip):
    """The premise of the project. If this fails, nothing else matters."""
    cfg = Config()
    base = run_baseline(long_hazard_clip, MockCortex(seed=2), cfg)
    mav = run_mavis(
        long_hazard_clip, MockCortex(seed=2), LocalMemory(path=None, load=False), cfg
    )
    assert mav.total_tokens < base.total_tokens


def test_short_clips_can_cost_more_not_less(hazard_clip):
    """An honest limit, pinned so it cannot be quietly claimed away.

    On a few seconds of footage MAVIS can cost *more* than a 1s-sampled baseline.
    The recall floor spends a strong call on the first gated frame and then scales
    its interval to the clip length, so short material carries the same fixed
    overhead with less redundancy to amortise it against — and MAVIS additionally
    pays for a cheap look before each escalation, which the baseline never does.

    The method needs sustained footage to pay off. Asserting only a loose upper
    bound here, rather than a saving, keeps that limitation in the test suite
    instead of in a footnote nobody reads.
    """
    cfg = Config()
    base = run_baseline(hazard_clip, MockCortex(seed=2), cfg)
    mav = run_mavis(hazard_clip, MockCortex(seed=2), LocalMemory(path=None, load=False), cfg)
    assert mav.total_tokens / base.total_tokens <= 1.3


def test_skip_costs_nothing(hazard_clip):
    cfg = Config()
    trace = run_mavis(hazard_clip, MockCortex(seed=3), LocalMemory(path=None, load=False), cfg)
    for step in trace.steps:
        if step.decision.action is Action.SKIP:
            assert step.cost is None


def test_learning_writes_episodes(hazard_clip):
    store = LocalMemory(path=None, load=False)
    run_mavis(hazard_clip, MockCortex(seed=4), store, Config(), learn=True)
    assert len(store) > 0


def test_learning_disabled_writes_nothing(hazard_clip):
    store = LocalMemory(path=None, load=False)
    run_mavis(hazard_clip, MockCortex(seed=4), store, Config(), learn=False)
    assert len(store) == 0


def test_outcome_source_is_validated(hazard_clip):
    with pytest.raises(ValueError):
        run_mavis(
            hazard_clip,
            MockCortex(seed=0),
            LocalMemory(path=None, load=False),
            Config(),
            outcome_source="oracle",
        )


# -- metrics -----------------------------------------------------------------


def test_recall_is_measured_per_clip_not_per_frame(hazard_clip, safe_clip):
    cfg = Config()
    cortex = MockCortex(seed=5)
    traces = [run_baseline(c, cortex, cfg) for c in (hazard_clip, safe_clip)]
    m = evaluate(traces, cfg.belief.detect_threshold)
    assert m.clips == 2
    assert m.hazard_clips == 1
    assert m.safe_clips == 1
    assert 0.0 <= m.recall <= 1.0


def test_benchmark_produces_a_comparison(hazard_clip, safe_clip, tmp_path):
    cfg = Config()
    result = benchmark.run(
        [hazard_clip, safe_clip],
        MockCortex(seed=6),
        LocalMemory(path=None, load=False),
        cfg,
    )
    assert result.comparison.baseline.clips == 2
    assert result.comparison.mavis.clips == 2
    # A mock run must be flagged so its numbers are never mistaken for evidence.
    assert result.comparison.estimated
    out = result.save(tmp_path / "bench.json")
    assert out.exists()


def test_benchmark_refuses_an_empty_clip_set():
    with pytest.raises(ValueError):
        benchmark.run([], MockCortex(), LocalMemory(path=None, load=False), Config())
