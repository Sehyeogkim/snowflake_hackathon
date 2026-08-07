"""The Cortex boundary.

Every AI inference in MAVIS goes through this one interface, and every inference
reports its own measured cost alongside its answer. Keeping cost on the same
return value as the result is deliberate: the scheduler's whole job is trading
one against the other, so they must never drift apart.

Two implementations exist:

* :class:`~mavis.cortex.snowflake.SnowflakeCortex` — the real thing.
  ``AI_CLASSIFY`` for the cheap look, ``AI_COMPLETE`` for strong reasoning,
  token counts from ``show_details => TRUE``.
* :class:`~mavis.cortex.mock.MockCortex` — development only. It fabricates
  plausible answers and token counts so the pipeline can be built and tested
  before a Snowflake account exists. Anything it produces is marked
  ``estimated=True`` and must never appear in a reported benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .. types import CostRecord, InferenceResult, SceneSummary


class CortexClient(Protocol):
    """What MAVIS needs from an inference backend, and nothing more."""

    name: str
    #: True when costs are fabricated rather than measured.
    estimated: bool
    #: Unit of the ``credits`` field on CostRecord — "credits" for Snowflake,
    #: "USD" for Gemini. Reported alongside the number so a benchmark table can
    #: never present one backend's unit as another's.
    cost_unit: str

    def classify(
        self, image: np.ndarray, categories: Sequence[str]
    ) -> tuple[SceneSummary, CostRecord]:
        """Cheap visual understanding of a single frame (AI_CLASSIFY)."""
        ...

    def complete(
        self, images: Sequence[np.ndarray], prompt: str, *, strong: bool
    ) -> tuple[InferenceResult, CostRecord]:
        """VLM reasoning over one or more frames (AI_COMPLETE).

        ``strong=False`` selects the cheaper model tier; ``strong=True`` the
        capable one. Passing more than one image is the MULTIFRAME action:
        temporal context for hazards that only exist across time.
        """
        ...

    def close(self) -> None: ...


@dataclass(slots=True)
class CostLedger:
    """Running total of everything spent, per model and per action tier."""

    records: list[CostRecord] = field(default_factory=list)

    def add(self, record: CostRecord) -> CostRecord:
        self.records.append(record)
        return record

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_credits(self) -> float:
        return sum(r.credits or 0.0 for r in self.records)

    @property
    def any_estimated(self) -> bool:
        return any(r.estimated for r in self.records)

    def by_model(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records:
            slot = out.setdefault(
                r.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "credits": 0.0}
            )
            slot["calls"] += 1
            slot["prompt_tokens"] += r.prompt_tokens
            slot["completion_tokens"] += r.completion_tokens
            slot["credits"] += r.credits or 0.0
        return out

    def query_ids(self) -> list[str]:
        """Snowflake query ids, for reconciling against ACCOUNT_USAGE later."""
        return [r.query_id for r in self.records if r.query_id]


#: The prompt that turns the strong tier into a hazard judge. Kept in one place
#: so baseline and MAVIS provably ask the model the same question — the benchmark
#: is only fair if the only difference between them is *when* they ask.
#:
#: The taxonomy is spelled out because "is this dangerous?" is not the question
#: the dataset labels answer. Asked generically, a VLM flags a forklift with a
#: view-obstructing stack as hazardous — a defensible safety observation, but
#: that clip is labelled *safe carrying*, and the label means "not an overload".
#: Naming the four labelled violations and their safe counterparts aligns the
#: question with the ground truth instead of grading the model against a
#: definition it was never given.
_HAZARD_TAXONOMY = """Report a hazard ONLY for one of these four specific conditions:

1. WALKWAY VIOLATION — a person walking or standing outside the marked pedestrian
   walkway. Look at the painted floor lines to decide.
2. UNAUTHORISED INTERVENTION — a person reaching into, leaning over, climbing on
   or servicing machinery that is running or has not been isolated.
3. OPEN PANEL COVER — an electrical panel or control cabinet whose cover is open,
   removed, or hanging loose, exposing the interior.
4. FORKLIFT OVERLOAD — a forklift carrying a load that is clearly oversized,
   unsecured or unstable for that truck.

Do NOT report a hazard for any of these:
- a person walking inside the marked walkway
- servicing performed with guards, isolation or permits in place
- a closed, intact panel cover
- a forklift carrying a normal secured load, even when it is stacked high or
  partly obstructs the driver's view
- ordinary factory activity, poor housekeeping, or anything not in the list above

Judge only what is visible. Do not speculate about what might happen next."""

HAZARD_PROMPT = f"""You are a factory safety inspector reviewing one CCTV frame.

{_HAZARD_TAXONOMY}

hazard_prob is your probability that one of the four conditions is present.
confidence is how sure you are of that judgement given image quality and framing."""

MULTIFRAME_PROMPT = f"""You are a factory safety inspector reviewing consecutive CCTV
frames from the same camera, in chronological order.

{_HAZARD_TAXONOMY}

Use the motion between frames as evidence: a worker crossing out of the walkway,
a load shifting, or a person moving into a machine's envelope is visible across
frames even when no single frame settles it.

hazard_prob is your probability that one of the four conditions occurs anywhere in
the sequence. confidence is how sure you are of that judgement."""
