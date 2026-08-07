from pathlib import Path

from mavis.manifest import assign_grouped_splits
from mavis.models import ClipRecord


def test_groups_never_cross_splits() -> None:
    records = [
        ClipRecord(str(i), Path(f"{i}.mp4"), "safe_walkway", group_id=f"g{i // 2}")
        for i in range(12)
    ]
    split = assign_grouped_splits(records, seed_count=4, dev_count=4)
    group_splits: dict[str, set[str]] = {}
    for record in split:
        group_splits.setdefault(record.group_id or "", set()).add(record.split)
    assert all(len(names) == 1 for names in group_splits.values())
