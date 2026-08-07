from __future__ import annotations

import json

from mavis.gemini_teacher import (
    GeminiTeacherStore,
    ModelPrice,
    build_experiences,
    choose_occlusion_window,
    prioritized_candidates,
)


def test_budget_reservation_never_crosses_cap(tmp_path) -> None:
    store = GeminiTeacherStore(tmp_path / "budget.sqlite3")
    store.add_prior_spend(0.25)
    assert store.reserve("a", "clip-a", "cheap", 0.50, 1.0)
    assert not store.reserve("b", "clip-b", "cheap", 0.26, 1.0)
    store.settle("a", 0.20, 100, 20)
    assert store.reserve("b", "clip-b", "cheap", 0.26, 1.0)
    assert store.budget() == {"spent": 0.45, "reserved": 0.26}


def test_cost_uses_public_per_million_rates() -> None:
    assert ModelPrice(2.0, 12.0).estimate(10_000, 500) == 0.026


def test_unsafe_false_negative_is_first_priority() -> None:
    scores = json.dumps({"safe_carrying": 0.9, "opened_panel_cover": 0.1})
    rows = [
        {
            "clip_id": "safe-hard",
            "label": "safe_walkway",
            "prediction": "opened_panel_cover",
            "scores_json": scores,
            "needs_temporal_context": 1,
        },
        {
            "clip_id": "unsafe-fn",
            "label": "opened_panel_cover",
            "prediction": "safe_carrying",
            "scores_json": scores,
            "needs_temporal_context": 0,
        },
    ]
    assert prioritized_candidates(rows)[0]["clip_id"] == "unsafe-fn"


def test_occlusion_window_is_bounded() -> None:
    start, end = choose_occlusion_window(
        [{"start_sec": 0, "end_sec": 10, "reason": "whole clip"}], 10
    )
    assert 0 <= start < end <= 10
    assert end - start <= 2.0


def test_every_cheap_run_becomes_ground_truth_evaluated_experience(tmp_path) -> None:
    store = GeminiTeacherStore(tmp_path / "teacher.sqlite3")
    proxy = tmp_path / "clip.mp4"
    proxy.write_bytes(b"not-read-for-cheap-only")
    store.save_run(
        {
            "clip_id": "clip-1",
            "stage": "cheap_dense",
            "split": "seed",
            "label": "opened_panel_cover",
            "source_path": str(proxy),
            "proxy_path": str(proxy),
            "model": "gemini-test",
            "fps": 10,
            "media_resolution": "low",
            "status": "success",
            "prediction": "closed_panel_cover",
            "scores_json": json.dumps({"opened_panel_cover": 0.1, "closed_panel_cover": 0.9}),
            "reported_unsafe": 0,
            "needs_temporal_context": 1,
            "scene": "panel",
            "windows_json": "[]",
            "evidence_json": "[]",
            "prompt_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "estimated_usd": 0.001,
            "latency_ms": 1,
            "error": None,
            "raw_json": "{}",
            "updated_at": "now",
        }
    )

    experiences = build_experiences(store)

    assert len(experiences) == 1
    payload = experiences[0]["payload"]
    assert payload["task_completed"] is True
    assert payload["outcome"] == "unsafe_escalation_rule_verified"
    assert payload["best_action"] == "abstain_and_escalate_unsafe_false_negative"
