from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .labels import CANONICAL_LABELS, UNSAFE_LABELS, normalize_label


class ProviderEvaluationError(ValueError):
    """Raised when provider benchmark rows cannot produce a trustworthy summary."""


@dataclass(frozen=True)
class ProviderBenchmarkFields:
    """Column names used to read a provider-comparison benchmark row.

    The default schema deliberately distinguishes measured provider usage from the
    estimated dense baseline::

        {
          "clip_id": "clip-001",
          "ground_truth": "opened_panel_cover",
          "actual_prediction": "opened_panel_cover",
          "reference_prediction": "opened_panel_cover",
          "actual_input_tokens": 120,
          "actual_output_tokens": 30,
          "actual_total_tokens": 150,
          "dense_baseline_tokens": 1200
        }

    ``actual_total_tokens`` may be omitted, in which case input + output is used.
    Custom field names make the evaluator reusable for any pair of providers.
    """

    clip_id: str = "clip_id"
    ground_truth: str = "ground_truth"
    actual_prediction: str = "actual_prediction"
    reference_prediction: str = "reference_prediction"
    actual_input_tokens: str = "actual_input_tokens"
    actual_output_tokens: str = "actual_output_tokens"
    actual_total_tokens: str = "actual_total_tokens"
    dense_baseline_tokens: str = "dense_baseline_tokens"


@dataclass(frozen=True)
class ProviderGateTargets:
    token_reduction_pct: float = 85.0
    provider_agreement_pct: float = 99.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ProviderEvaluationError(f"{name} must be between 0 and 100")


@dataclass(frozen=True)
class RateMetric:
    numerator: int
    denominator: int
    pct: float | None

    @classmethod
    def from_counts(cls, numerator: int, denominator: int) -> RateMetric:
        pct = None if denominator == 0 else 100.0 * numerator / denominator
        return cls(numerator=numerator, denominator=denominator, pct=pct)


@dataclass(frozen=True)
class GateResult:
    actual_pct: float
    target_pct: float
    passed: bool


@dataclass(frozen=True)
class ProviderBenchmarkSummary:
    row_count: int
    actual_input_tokens: int
    actual_output_tokens: int
    actual_total_tokens: int
    dense_baseline_tokens: int
    token_reduction_pct: float
    provider_agreement: RateMetric
    ground_truth_accuracy: RateMetric
    unsafe_recall: RateMetric
    token_reduction_gate: GateResult
    provider_agreement_gate: GateResult
    gates_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "tokens": {
                "actual_input_tokens": self.actual_input_tokens,
                "actual_output_tokens": self.actual_output_tokens,
                "actual_total_tokens": self.actual_total_tokens,
                "dense_baseline_tokens": self.dense_baseline_tokens,
                "token_reduction_pct": self.token_reduction_pct,
            },
            "quality": {
                "provider_agreement": asdict(self.provider_agreement),
                "ground_truth_accuracy": asdict(self.ground_truth_accuracy),
                "unsafe_recall": asdict(self.unsafe_recall),
            },
            "gates": {
                "token_reduction": asdict(self.token_reduction_gate),
                "provider_agreement": asdict(self.provider_agreement_gate),
                "passed": self.gates_passed,
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    def to_markdown(self, *, title: str = "Provider / Token Benchmark Summary") -> str:
        status = lambda passed: "PASS" if passed else "FAIL"
        unsafe = _format_pct(self.unsafe_recall.pct)
        accuracy = _format_pct(self.ground_truth_accuracy.pct)
        agreement = _format_pct(self.provider_agreement.pct)
        rows = [
            f"# {title}",
            "",
            f"Rows evaluated: **{self.row_count:,}**",
            "",
            "| Metric | Result | Target | Status |",
            "|---|---:|---:|:---:|",
            f"| Actual input tokens | {self.actual_input_tokens:,} | — | — |",
            f"| Actual output tokens | {self.actual_output_tokens:,} | — | — |",
            f"| Actual total tokens | {self.actual_total_tokens:,} | — | — |",
            f"| Dense baseline tokens | {self.dense_baseline_tokens:,} | — | — |",
            (
                f"| Token reduction | {_format_pct(self.token_reduction_pct)} | "
                f">= {_format_pct(self.token_reduction_gate.target_pct)} | "
                f"{status(self.token_reduction_gate.passed)} |"
            ),
            (
                f"| Provider agreement | {agreement} "
                f"({self.provider_agreement.numerator}/{self.provider_agreement.denominator}) | "
                f">= {_format_pct(self.provider_agreement_gate.target_pct)} | "
                f"{status(self.provider_agreement_gate.passed)} |"
            ),
            (
                f"| Ground-truth accuracy | {accuracy} "
                f"({self.ground_truth_accuracy.numerator}/"
                f"{self.ground_truth_accuracy.denominator}) | — | — |"
            ),
            (
                f"| Unsafe recall | {unsafe} "
                f"({self.unsafe_recall.numerator}/{self.unsafe_recall.denominator}) | — | — |"
            ),
            "",
            f"Overall target gates: **{status(self.gates_passed)}**",
        ]
        return "\n".join(rows) + "\n"

    def write_json(self, path: Path, *, indent: int | None = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    def write_markdown(
        self, path: Path, *, title: str = "Provider / Token Benchmark Summary"
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(title=title), encoding="utf-8")


@dataclass(frozen=True)
class _NormalizedRow:
    clip_id: str
    ground_truth: str
    actual_prediction: str
    reference_prediction: str
    actual_input_tokens: int
    actual_output_tokens: int
    actual_total_tokens: int
    dense_baseline_tokens: int


def _format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _required(row: Mapping[str, Any], field: str, row_index: int) -> Any:
    if field not in row or row[field] is None:
        raise ProviderEvaluationError(f"row {row_index}: missing required field {field!r}")
    return row[field]


def _text(value: Any, field: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderEvaluationError(f"row {row_index}: {field!r} must be a non-empty string")
    return value.strip()


def _token_count(value: Any, field: str, row_index: int) -> int:
    if isinstance(value, bool):
        raise ProviderEvaluationError(f"row {row_index}: {field!r} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise ProviderEvaluationError(f"row {row_index}: {field!r} must be an integer")
    if result < 0:
        raise ProviderEvaluationError(f"row {row_index}: {field!r} cannot be negative")
    return result


def _normalize_row(
    row: Mapping[str, Any], fields: ProviderBenchmarkFields, row_index: int
) -> _NormalizedRow:
    clip_id = _text(_required(row, fields.clip_id, row_index), fields.clip_id, row_index)
    ground_truth = normalize_label(
        _text(_required(row, fields.ground_truth, row_index), fields.ground_truth, row_index)
    )
    if ground_truth not in CANONICAL_LABELS:
        raise ProviderEvaluationError(
            f"row {row_index}: ground truth {ground_truth!r} is not a canonical label"
        )
    actual_prediction = normalize_label(
        _text(
            _required(row, fields.actual_prediction, row_index),
            fields.actual_prediction,
            row_index,
        )
    )
    reference_prediction = normalize_label(
        _text(
            _required(row, fields.reference_prediction, row_index),
            fields.reference_prediction,
            row_index,
        )
    )
    input_tokens = _token_count(
        _required(row, fields.actual_input_tokens, row_index),
        fields.actual_input_tokens,
        row_index,
    )
    output_tokens = _token_count(
        _required(row, fields.actual_output_tokens, row_index),
        fields.actual_output_tokens,
        row_index,
    )
    raw_total = row.get(fields.actual_total_tokens)
    total_tokens = (
        input_tokens + output_tokens
        if raw_total is None
        else _token_count(raw_total, fields.actual_total_tokens, row_index)
    )
    if total_tokens < input_tokens + output_tokens:
        raise ProviderEvaluationError(
            f"row {row_index}: {fields.actual_total_tokens!r} cannot be less than "
            "input + output tokens"
        )
    baseline_tokens = _token_count(
        _required(row, fields.dense_baseline_tokens, row_index),
        fields.dense_baseline_tokens,
        row_index,
    )
    return _NormalizedRow(
        clip_id=clip_id,
        ground_truth=ground_truth,
        actual_prediction=actual_prediction,
        reference_prediction=reference_prediction,
        actual_input_tokens=input_tokens,
        actual_output_tokens=output_tokens,
        actual_total_tokens=total_tokens,
        dense_baseline_tokens=baseline_tokens,
    )


def evaluate_provider_benchmark(
    rows: Iterable[Mapping[str, Any]],
    *,
    fields: ProviderBenchmarkFields | None = None,
    targets: ProviderGateTargets | None = None,
    unsafe_labels: Iterable[str] = UNSAFE_LABELS,
) -> ProviderBenchmarkSummary:
    """Aggregate measured token usage and provider quality from benchmark rows.

    Provider agreement compares normalized ``actual_prediction`` and
    ``reference_prediction`` values. Ground-truth accuracy and unsafe recall are
    evaluated for ``actual_prediction``. Unsafe recall is binary safety recall:
    an unsafe ground-truth row is recalled when the actual prediction is any
    unsafe class. It is reported as ``None`` when the input has no unsafe rows.
    """

    fields = fields or ProviderBenchmarkFields()
    targets = targets or ProviderGateTargets()
    normalized = [_normalize_row(row, fields, index) for index, row in enumerate(rows)]
    if not normalized:
        raise ProviderEvaluationError("at least one benchmark row is required")
    clip_ids = [row.clip_id for row in normalized]
    if len(set(clip_ids)) != len(clip_ids):
        raise ProviderEvaluationError("clip_id values must be unique")

    unsafe = frozenset(normalize_label(label) for label in unsafe_labels)
    actual_input = sum(row.actual_input_tokens for row in normalized)
    actual_output = sum(row.actual_output_tokens for row in normalized)
    actual_total = sum(row.actual_total_tokens for row in normalized)
    dense_total = sum(row.dense_baseline_tokens for row in normalized)
    if dense_total <= 0:
        raise ProviderEvaluationError("dense baseline token total must be greater than zero")

    agreement_count = sum(
        row.actual_prediction == row.reference_prediction for row in normalized
    )
    correct_count = sum(row.actual_prediction == row.ground_truth for row in normalized)
    unsafe_rows = [row for row in normalized if row.ground_truth in unsafe]
    unsafe_recalled = sum(row.actual_prediction in unsafe for row in unsafe_rows)

    token_reduction = 100.0 * (1.0 - actual_total / dense_total)
    agreement = RateMetric.from_counts(agreement_count, len(normalized))
    accuracy = RateMetric.from_counts(correct_count, len(normalized))
    recall = RateMetric.from_counts(unsafe_recalled, len(unsafe_rows))
    assert agreement.pct is not None
    token_gate = GateResult(
        actual_pct=token_reduction,
        target_pct=targets.token_reduction_pct,
        passed=token_reduction >= targets.token_reduction_pct,
    )
    agreement_gate = GateResult(
        actual_pct=agreement.pct,
        target_pct=targets.provider_agreement_pct,
        passed=agreement.pct >= targets.provider_agreement_pct,
    )
    return ProviderBenchmarkSummary(
        row_count=len(normalized),
        actual_input_tokens=actual_input,
        actual_output_tokens=actual_output,
        actual_total_tokens=actual_total,
        dense_baseline_tokens=dense_total,
        token_reduction_pct=token_reduction,
        provider_agreement=agreement,
        ground_truth_accuracy=accuracy,
        unsafe_recall=recall,
        token_reduction_gate=token_gate,
        provider_agreement_gate=agreement_gate,
        gates_passed=token_gate.passed and agreement_gate.passed,
    )
