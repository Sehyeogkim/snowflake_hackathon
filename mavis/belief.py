"""Hazard belief state and the information-gain accounting built on it.

The belief is a single Bernoulli probability that the scene currently on camera
contains a hazard. It is maintained in log-odds so that observations of differing
strength compose additively, and so that a strong VLM verdict can outweigh a
cheap classifier without any special-casing.

Information gain is the honest, measured quantity that MAVIS is optimising:

    ΔH = H(b_before) - H(b_after)

in bits. It is computed *after* a call returns, never predicted and then assumed.
The predicted value used for scheduling comes from memory (see
:mod:`mavis.scheduler`); this module only ever reports what actually happened.
"""

from __future__ import annotations

import math

from .config import BeliefConfig
from .types import Action, InferenceResult

_EPS = 1e-6


def entropy(p: float) -> float:
    """Binary entropy in bits."""
    p = min(max(p, _EPS), 1.0 - _EPS)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HazardBelief:
    """Running belief that a hazard is present, updated by inference results."""

    def __init__(self, cfg: BeliefConfig):
        self.cfg = cfg
        self._logit = logit(cfg.prior)
        self._prior_logit = logit(cfg.prior)
        self._last_ts: float | None = None
        self.peak: float = cfg.prior

    @property
    def p(self) -> float:
        return sigmoid(self._logit)

    @property
    def entropy(self) -> float:
        return entropy(self.p)

    @property
    def detected(self) -> bool:
        return self.p >= self.cfg.detect_threshold

    def decay_to(self, timestamp_s: float) -> None:
        """Pull the belief back toward the prior for elapsed unobserved time.

        Without this, a hazard confirmed once would keep the belief pinned high
        for the rest of the clip and SKIP would look free forever.
        """
        if self._last_ts is None:
            self._last_ts = timestamp_s
            return
        dt = max(timestamp_s - self._last_ts, 0.0)
        self._last_ts = timestamp_s
        if dt <= 0:
            return
        pull = 1.0 - math.exp(-self.cfg.decay_per_s * dt)
        self._logit += (self._prior_logit - self._logit) * pull

    def observe(
        self, result: InferenceResult, action: Action, timestamp_s: float
    ) -> float:
        """Fold in an inference result. Returns the realised information gain.

        The observation's pull is scaled by both the action's intrinsic weight
        (a strong VLM is trusted more than a cheap classifier) and the model's
        own stated confidence.
        """
        self.decay_to(timestamp_s)
        before = self.entropy

        weight = self.cfg.weights.get(action, 1.0) * max(result.confidence, _EPS)
        # Observation expressed as a log-odds nudge relative to the prior, so a
        # result that merely restates the prior moves nothing.
        self._logit += weight * (logit(result.hazard_prob) - self._prior_logit)

        after = self.entropy
        self.peak = max(self.peak, self.p)
        return before - after

    def counterfactual_gain(self, hazard_prob: float, action: Action) -> float:
        """What ΔH *would* be for a hypothetical result, without mutating state.

        Used to sanity-check the scheduler's expectations in tests; the live path
        never relies on it.
        """
        before = self.entropy
        weight = self.cfg.weights.get(action, 1.0)
        shifted = self._logit + weight * (logit(hazard_prob) - self._prior_logit)
        return before - entropy(sigmoid(shifted))
