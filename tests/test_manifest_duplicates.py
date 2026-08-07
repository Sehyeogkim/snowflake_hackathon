from pathlib import Path

from mavis.manifest import assign_grouped_splits, merge_duplicate_groups
from mavis.models import ClipRecord


def clip(clip_id: str, label: str = "safe_walkway") -> ClipRecord:
    return ClipRecord(clip_id, Path(f"{clip_id}.mp4"), label, group_id=f"clip:{clip_id}")


def test_exact_duplicate_pair_cannot_cross_splits() -> None:
    records = [clip("a"), clip("b"), clip("c"), clip("d")]
    merged = merge_duplicate_groups(
        records,
        [
            {
                "left_clip_id": "a",
                "right_clip_id": "b",
                "mean_hamming": 0.0,
                "duration_delta_s": 0.0,
            }
        ],
    )
    assigned = assign_grouped_splits(merged, seed_count=1, dev_count=1)
    split = {row.clip_id: row.split for row in assigned}
    assert split["a"] == split["b"]


def test_near_match_is_not_merged_under_exact_defaults() -> None:
    records = [clip("a"), clip("b")]
    merged = merge_duplicate_groups(
        records,
        [
            {
                "left_clip_id": "a",
                "right_clip_id": "b",
                "mean_hamming": 1.0,
                "duration_delta_s": 0.0,
            }
        ],
    )
    assert merged[0].group_id != merged[1].group_id


def test_forced_seed_keeps_its_whole_group_and_preserves_test_target() -> None:
    records = [clip(str(index)) for index in range(8)]
    records[0] = ClipRecord("0", Path("0.mp4"), "safe_walkway", group_id="pair")
    records[1] = ClipRecord("1", Path("1.mp4"), "safe_walkway", group_id="pair")
    assigned = assign_grouped_splits(
        records,
        seed_count=1,
        dev_count=2,
        test_count=3,
        forced_seed_clip_ids=frozenset({"0"}),
    )
    splits = {row.clip_id: row.split for row in assigned}
    assert splits["0"] == splits["1"] == "seed"
    assert sum(row.split == "test" for row in assigned) == 3
