"""Development-only stand-in for Snowflake Cortex.

DO NOT REPORT NUMBERS PRODUCED BY THIS CLIENT. Every CostRecord it returns is
flagged ``estimated=True`` and :mod:`mavis.benchmark` refuses to print a headline
cost-reduction figure from an estimated run. Its purpose is to exercise the full
pipeline — gate, memory, scheduler, belief, ledger — before a Snowflake account
exists, so that swapping in :class:`~mavis.cortex.snowflake.SnowflakeCortex` is a
one-line change rather than an integration project.

The simulated model is deliberately *not* an oracle. It reproduces the property
that makes this dataset interesting: the cheap tier confuses each unsafe class
with its safe counterpart, and only the strong tier separates them. If MAVIS
looked good against a mock that made the cheap tier accurate, the result would
mean nothing.
"""

from __future__ import annotations

import numpy as np

from ..dataset import CLASSES
from ..types import CostRecord, InferenceResult, SceneSummary
from .base import CortexClient  # noqa: F401  (documents the implemented protocol)

#: Placeholder credit rates, credits per 1M tokens. Replaced wholesale by
#: CORTEX_AI_FUNCTIONS_USAGE_HISTORY once a real account is connected.
MOCK_CREDIT_RATE = {
    "mock-classify": 0.30,
    "mock-cheap": 0.55,
    "mock-strong": 2.55,
}

#: Discrimination power of each tier: probability the model's verdict points the
#: right way for a confusable safe/unsafe pair.
_TIER_SKILL = {"mock-classify": 0.58, "mock-cheap": 0.72, "mock-strong": 0.93}


class MockCortex:
    """Simulated Cortex with plausible answers, latencies and token counts."""

    name = "mock"
    estimated = True
    cost_unit = "fake"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._truth_class: int | None = None

    # -- test harness hook -------------------------------------------------

    def set_scene_truth(self, class_id: int | None) -> None:
        """Tell the simulator which clip it is looking at.

        Only the mock has this; the real client obviously never sees a label.
        The runner calls it per clip and the scheduler never reads it.
        """
        self._truth_class = class_id

    # -- CortexClient ------------------------------------------------------

    def classify(self, image, categories):
        tokens_in = self._image_tokens(image) + 60
        cost = self._cost("mock-classify", tokens_in, 12)

        hazard = self._noisy_hazard("mock-classify")
        risk = float(np.clip(self.rng.normal(0.62 if hazard else 0.34, 0.13), 0.02, 0.98))
        label, objects = self._scene_content()
        return (
            SceneSummary(
                labels=[label],
                objects=objects,
                risk=risk,
                raw={"simulated": True, "categories": list(categories)[:3]},
            ),
            cost,
        )

    def complete(self, images, prompt: str, *, strong: bool):
        model = "mock-strong" if strong else "mock-cheap"
        tokens_in = sum(self._image_tokens(im) for im in images) + len(prompt) // 4
        tokens_out = int(self.rng.integers(48, 96))
        cost = self._cost(model, tokens_in, tokens_out)

        hazard = self._noisy_hazard(model, multiframe=len(images) > 1)
        centre = 0.82 if hazard else 0.16
        prob = float(np.clip(self.rng.normal(centre, 0.10), 0.02, 0.98))
        confidence = float(np.clip(self.rng.normal(0.80 if strong else 0.58, 0.08), 0.1, 0.99))
        desc = CLASSES[self._truth_class][2] if self._truth_class is not None else "scene"
        return (
            InferenceResult(
                hazard_prob=prob,
                confidence=confidence,
                rationale=f"[simulated] frame is consistent with {desc}",
                evidence=["simulated evidence"],
                raw={"simulated": True},
            ),
            cost,
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass

    # -- internals ---------------------------------------------------------

    def _noisy_hazard(self, model: str, multiframe: bool = False) -> bool:
        """Simulate a verdict whose accuracy depends on the model tier."""
        if self._truth_class is None:
            return bool(self.rng.random() < 0.3)
        truth = CLASSES[self._truth_class][1]
        skill = _TIER_SKILL[model] + (0.04 if multiframe else 0.0)
        return truth if self.rng.random() < skill else not truth

    def _scene_content(self) -> tuple[str, list[str]]:
        """Scene labels that a cheap classifier could plausibly emit.

        Crucially these are shared between a class and its safe counterpart —
        the cheap tier sees 'worker near forklift', not 'violation'.
        """
        pools = {
            0: ("worker on plant floor", ["worker", "walkway marking", "floor"]),
            1: ("person at machinery", ["worker", "machine", "control panel"]),
            2: ("electrical panel in view", ["panel", "cabling", "enclosure"]),
            3: ("forklift handling a load", ["forklift", "pallet", "load"]),
        }
        if self._truth_class is None:
            return "unidentified scene", ["unknown"]
        base = self._truth_class if self._truth_class < 4 else CLASSES[self._truth_class][3]
        label, objects = pools[base]
        return label, list(objects)

    @staticmethod
    def _image_tokens(image: np.ndarray) -> int:
        """Rough vision-token count: proportional to pixel area, as real VLMs are."""
        h, w = image.shape[:2]
        return max(int(h * w / 750), 32)

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> CostRecord:
        total = tokens_in + tokens_out
        return CostRecord(
            model=model,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            credits=total / 1e6 * MOCK_CREDIT_RATE[model],
            query_id=None,
            latency_s=float(self.rng.uniform(0.05, 0.2)),
            estimated=True,
        )
