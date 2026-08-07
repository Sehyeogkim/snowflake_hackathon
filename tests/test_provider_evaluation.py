from __future__ import annotations

import json

import pytest

from mavis.provider_evaluation import (
    ProviderBenchmarkFields,
    ProviderEvaluationError,
    ProviderGateTargets,
    evaluate_provider_benchmark,
)


def _row(
    clip_id: str,
    gt: str,
    actual: str,
    reference: str,
    input_tokens: int = 80,
    output_tokens: int = 20,
    total_tokens: int | None = 100,
    dense_tokens: int = 1_000,
) -> dict[str, object]:
    row: dict[str, object] = {
        "clip_id": clip_id,
        "ground_truth": gt,
        "actual_prediction": actual,
        "reference_prediction": reference,
        "actual_input_tokens": input_tokens,
        "actual_output_tokens": output_tokens,
        "dense_baseline_tokens": dense_tokens,
    }
    if total_tokens is not None:
        row["actual_total_tokens"] = total_tokens
    return row


def test_summary_aggregates_tokens_quality_and_default_gates() -> None:
    rows = [
        _row("a", "Opened Panel Cover", "opened_panel_cover", "Opened Panel Cover"),
        _row("b", "safe_walkway", "safe_walkway", "safe_walkway"),
        _row(
            "c",
            "unauthorized_intervention",
            "safe_walkway",
            "unauthorized_intervention",
        ),
    ]

    summary = evaluate_provider_benchmark(rows)

    assert summary.actual_input_tokens == 240
    assert summary.actual_output_tokens == 60
    assert summary.actual_total_tokens == 300
    assert summary.dense_baseline_tokens == 3_000
    assert summary.token_reduction_pct == pytest.approx(90.0)
    assert summary.provider_agreement.numerator == 2
    assert summary.provider_agreement.pct == pytest.approx(200 / 3)
    assert summary.ground_truth_accuracy.pct == pytest.approx(200 / 3)
    assert summary.unsafe_recall.numerator == 1
    assert summary.unsafe_recall.denominator == 2
    assert summary.unsafe_recall.pct == pytest.approx(50.0)
    assert summary.token_reduction_gate.passed is True
    assert summary.provider_agreement_gate.passed is False
    assert summary.gates_passed is False


def test_thresholds_are_inclusive_at_85_and_99_percent() -> None:
    rows = [
        _row(
            str(index),
            "safe_walkway",
            "safe_walkway",
            "safe_walkway" if index else "closed_panel_cover",
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            dense_tokens=1_000,
        )
        for index in range(100)
    ]

    summary = evaluate_provider_benchmark(rows)

    assert summary.token_reduction_pct == pytest.approx(85.0)
    assert summary.provider_agreement.pct == pytest.approx(99.0)
    assert summary.gates_passed is True


def test_total_defaults_to_input_plus_output_and_custom_fields_work() -> None:
    fields = ProviderBenchmarkFields(
        ground_truth="label",
        actual_prediction="snowflake_prediction",
        reference_prediction="gemini_prediction",
        actual_input_tokens="input_tokens",
        actual_output_tokens="output_tokens",
        actual_total_tokens="total_tokens",
        dense_baseline_tokens="dense_tokens",
    )
    row = {
        "clip_id": "clip-1",
        "label": "01 - Safe Walkway Violation",
        "snowflake_prediction": "safe_walkway_violation",
        "gemini_prediction": "Safe Walkway Violation",
        "input_tokens": "10",
        "output_tokens": 5.0,
        "dense_tokens": 100,
    }

    summary = evaluate_provider_benchmark([row], fields=fields)

    assert summary.actual_total_tokens == 15
    assert summary.token_reduction_pct == pytest.approx(85.0)
    assert summary.provider_agreement.pct == 100.0
    assert summary.ground_truth_accuracy.pct == 100.0
    assert summary.unsafe_recall.pct == 100.0


def test_json_markdown_and_file_outputs_are_stable(tmp_path) -> None:
    summary = evaluate_provider_benchmark(
        [_row("a", "safe_walkway", "safe_walkway", "safe_walkway")]
    )

    payload = json.loads(summary.to_json())
    markdown = summary.to_markdown(title="Cross-provider audit")
    json_path = tmp_path / "nested" / "summary.json"
    markdown_path = tmp_path / "nested" / "summary.md"
    summary.write_json(json_path)
    summary.write_markdown(markdown_path, title="Cross-provider audit")

    assert payload["tokens"]["token_reduction_pct"] == 90.0
    assert payload["gates"]["passed"] is True
    assert "# Cross-provider audit" in markdown
    assert "| Provider agreement | 100.00% (1/1) | >= 99.00% | PASS |" in markdown
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert markdown_path.read_text(encoding="utf-8") == markdown


def test_no_unsafe_ground_truth_reports_na_recall() -> None:
    summary = evaluate_provider_benchmark(
        [_row("a", "safe_walkway", "safe_walkway", "safe_walkway")]
    )

    assert summary.unsafe_recall.denominator == 0
    assert summary.unsafe_recall.pct is None
    assert "| Unsafe recall | N/A (0/0) |" in summary.to_markdown()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "at least one"),
        ([_row("a", "unknown", "unknown", "unknown")], "canonical label"),
        (
            [
                _row("a", "safe_walkway", "safe_walkway", "safe_walkway"),
                _row("a", "safe_walkway", "safe_walkway", "safe_walkway"),
            ],
            "unique",
        ),
        (
            [
                _row(
                    "a",
                    "safe_walkway",
                    "safe_walkway",
                    "safe_walkway",
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=14,
                )
            ],
            r"input \+ output",
        ),
        (
            [
                _row(
                    "a",
                    "safe_walkway",
                    "safe_walkway",
                    "safe_walkway",
                    dense_tokens=0,
                )
            ],
            "greater than zero",
        ),
    ],
)
def test_invalid_rows_fail_loudly(rows, message: str) -> None:
    with pytest.raises(ProviderEvaluationError, match=message):
        evaluate_provider_benchmark(rows)


def test_custom_gate_targets() -> None:
    summary = evaluate_provider_benchmark(
        [_row("a", "safe_walkway", "safe_walkway", "safe_walkway")],
        targets=ProviderGateTargets(token_reduction_pct=95, provider_agreement_pct=100),
    )

    assert summary.token_reduction_gate.passed is False
    assert summary.provider_agreement_gate.passed is True
    assert summary.gates_passed is False


def test_invalid_gate_target_is_rejected() -> None:
    with pytest.raises(ProviderEvaluationError, match="between 0 and 100"):
        ProviderGateTargets(token_reduction_pct=101)
