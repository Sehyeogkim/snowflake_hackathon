"""The belief update is where information gain becomes a number, so the
properties it must satisfy are worth pinning down: gain is only positive when an
observation actually reduces uncertainty, a strong verdict must outweigh a cheap
one, and confidence must be able to reach the detection threshold — otherwise
recall would be unreachable regardless of how well the scheduler behaved.
"""

from __future__ import annotations

import math

import pytest

from mavis.belief import HazardBelief, entropy
from mavis.config import BeliefConfig
from mavis.types import Action, InferenceResult


def result(p: float, conf: float = 0.9) -> InferenceResult:
    return InferenceResult(hazard_prob=p, confidence=conf)


def test_entropy_is_maximal_at_one_half():
    assert entropy(0.5) == pytest.approx(1.0)
    assert entropy(0.01) < 0.1
    assert entropy(0.99) < 0.1


def test_confident_observation_reduces_entropy():
    belief = HazardBelief(BeliefConfig())
    gain = belief.observe(result(0.95), Action.STRONG_VLM, timestamp_s=0.0)
    assert gain > 0
    assert belief.p > BeliefConfig().prior


def test_observation_matching_the_prior_yields_almost_no_gain():
    cfg = BeliefConfig()
    belief = HazardBelief(cfg)
    gain = belief.observe(result(cfg.prior, conf=0.9), Action.STRONG_VLM, timestamp_s=0.0)
    assert abs(gain) < 1e-6


def test_strong_moves_belief_further_than_cheap_for_the_same_verdict():
    cfg = BeliefConfig()
    cheap, strong = HazardBelief(cfg), HazardBelief(cfg)
    cheap.observe(result(0.9), Action.CLASSIFY, 0.0)
    strong.observe(result(0.9), Action.STRONG_VLM, 0.0)
    assert strong.p > cheap.p


def test_belief_can_cross_the_detection_threshold():
    # If a single confident strong call could not reach the threshold, no policy
    # could ever score a detection and the benchmark would be meaningless.
    cfg = BeliefConfig()
    belief = HazardBelief(cfg)
    belief.observe(result(0.95, conf=0.9), Action.STRONG_VLM, 0.0)
    assert belief.detected


def test_belief_decays_back_toward_the_prior_when_unobserved():
    cfg = BeliefConfig()
    belief = HazardBelief(cfg)
    belief.observe(result(0.95), Action.STRONG_VLM, 0.0)
    raised = belief.p
    belief.decay_to(30.0)
    assert belief.p < raised
    assert belief.p > cfg.prior  # decays toward, not instantly to


def test_peak_is_retained_across_decay():
    belief = HazardBelief(BeliefConfig())
    belief.observe(result(0.95), Action.STRONG_VLM, 0.0)
    peak = belief.peak
    belief.decay_to(60.0)
    assert belief.peak == peak


def test_each_gain_is_bounded_by_the_entropy_available_at_the_time():
    """A single observation cannot reveal more than was uncertain when it ran.

    The *cumulative* gain across a clip deliberately may exceed the starting
    entropy: belief decays back toward the prior between observations, so
    uncertainty returns and has to be paid for again. That is the behaviour that
    makes re-observing a long clip necessary rather than wasteful.
    """
    belief = HazardBelief(BeliefConfig())
    for t in range(5):
        # observe() applies elapsed-time decay before measuring, so the entropy
        # that was actually available is the post-decay value. decay_to is a
        # no-op when called twice for the same timestamp.
        belief.decay_to(float(t))
        available = belief.entropy
        gain = belief.observe(result(0.99), Action.STRONG_VLM, float(t))
        assert gain <= available + 1e-9
        assert math.isfinite(gain)


def test_cumulative_gain_can_exceed_starting_entropy_because_belief_decays():
    cfg = BeliefConfig()
    belief = HazardBelief(cfg)
    start = belief.entropy
    total = sum(
        belief.observe(result(0.99 if t % 2 == 0 else 0.01), Action.STRONG_VLM, float(t * 10))
        for t in range(6)
    )
    assert total > start
