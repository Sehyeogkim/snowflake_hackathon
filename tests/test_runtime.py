from pathlib import Path

from mavis.backends.mock import MockInferenceBackend
from mavis.config import Settings
from mavis.labels import CANONICAL_LABELS, UNSAFE_LABELS
from mavis.models import ClipRecord
from mavis.runtime import NoMemoryRetriever, RuntimeBenchmark


def settings() -> Settings:
    return Settings(
        labels=CANONICAL_LABELS,
        unsafe_labels=UNSAFE_LABELS,
        cheap_model="cheap",
        strong_model="strong",
        analysis_fps=2.0,
        scan_width=64,
        jpeg_quality=70,
        max_seed_clips=300,
        max_workers=1,
        actions=("cheap_single", "strong_single", "strong_multi"),
        crop_strong=False,
        prompt_version="test",
        safety_recall_tolerance_pp=2.0,
    )


def test_runtime_query_paths_are_separate(tmp_path: Path, monkeypatch) -> None:
    from mavis import runtime
    from mavis.models import CandidateFrame, CandidateSet

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"test")
    candidate = CandidateFrame("peak", 1, 0.1, image)
    candidates = CandidateSet("clip", 24, 3, 0.125, candidate, candidate, candidate)
    monkeypatch.setattr(runtime, "extract_candidates", lambda *args, **kwargs: candidates)
    clip = ClipRecord("clip", tmp_path / "clip.mp4", "safe_walkway", "dev")
    runner = RuntimeBenchmark(
        settings(),
        lambda: MockInferenceBackend("cheap", "strong", seed=1),
        lambda: MockInferenceBackend("cheap", "strong", seed=1),
        NoMemoryRetriever(),
        tmp_path,
        "test-run",
        workers=1,
    )
    report = runner.run([clip], tmp_path / "benchmark.jsonl")
    assert len(report.rows) == 1
    assert report.metrics["clips"] == 1
    assert report.metrics["claim_ready"] is False
    assert (tmp_path / "benchmark.jsonl").exists()
