from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from .labels import infer_label_from_path
from .models import ClipRecord

VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"})


def stable_id(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix().casefold()
    return hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]


def infer_group_id(path: Path) -> str:
    """Create a conservative leakage group from filename/camera/session hints."""
    stem = path.stem.casefold()
    original_stem = stem
    stem = re.sub(r"(?:clip|video|frame)?[_-]?\d+$", "", stem)
    tokens = [token for token in re.split(r"[_\-\s]+", stem) if token]
    camera_or_date = [
        token
        for token in tokens
        if re.fullmatch(r"(?:cam(?:era)?\d+|\d{4}(?:\d{2}){1,2}|session\d+)", token)
    ]
    if camera_or_date:
        return "-".join(camera_or_date)
    # Do not merge clips just because they share a class directory. A later
    # metadata/pHash audit can merge truly related clips conservatively.
    return f"clip:{path.parent.name.casefold()}:{original_stem}"


def discover_clips(root: Path) -> list[ClipRecord]:
    root = root.resolve()
    rows: list[ClipRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in VIDEO_SUFFIXES:
            continue
        label = infer_label_from_path(path)
        if not label:
            continue
        rows.append(
            ClipRecord(
                clip_id=stable_id(path, root),
                path=path.resolve(),
                label=label,
                group_id=infer_group_id(path),
                size_bytes=path.stat().st_size,
            )
        )
    return rows


def merge_duplicate_groups(
    records: Iterable[ClipRecord],
    pairs: Iterable[dict[str, object]],
    max_mean_hamming: float = 0.0,
    max_duration_delta_s: float = 0.0,
) -> list[ClipRecord]:
    """Merge verified fingerprint matches before assigning any split."""
    rows = list(records)
    by_id = {row.clip_id: row for row in rows}
    parent = {row.clip_id: row.clip_id for row in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        left = str(pair.get("left_clip_id", ""))
        right = str(pair.get("right_clip_id", ""))
        if left not in by_id or right not in by_id:
            continue
        if by_id[left].label != by_id[right].label:
            continue
        if float(pair.get("mean_hamming", float("inf"))) > max_mean_hamming:
            continue
        if float(pair.get("duration_delta_s", float("inf"))) > max_duration_delta_s:
            continue
        union(left, right)

    components: dict[str, list[str]] = {}
    for clip_id in parent:
        components.setdefault(find(clip_id), []).append(clip_id)
    duplicate_group = {}
    for members in components.values():
        if len(members) < 2:
            continue
        identity = ":".join(sorted(members))
        group_id = f"duplicate:{hashlib.sha1(identity.encode()).hexdigest()[:16]}"
        duplicate_group.update({member: group_id for member in members})

    return [
        ClipRecord(
            clip_id=row.clip_id,
            path=row.path,
            label=row.label,
            split="unassigned",
            group_id=duplicate_group.get(row.clip_id, row.group_id),
            size_bytes=row.size_bytes,
        )
        for row in rows
    ]


def assign_grouped_splits(
    records: Iterable[ClipRecord],
    seed_count: int = 300,
    dev_count: int = 91,
    random_seed: int = 42,
    test_count: int | None = None,
    forced_seed_clip_ids: frozenset[str] = frozenset(),
) -> list[ClipRecord]:
    """Assign whole groups to seed/dev/test while keeping class coverage balanced."""
    import random

    rng = random.Random(random_seed)
    grouped: dict[str, list[ClipRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_id or record.clip_id, []).append(record)

    groups = list(grouped.values())
    random_tie = {id(group): rng.random() for group in groups}
    groups.sort(key=lambda group: (-len(group), random_tie[id(group)]))
    total = sum(len(group) for group in groups)
    forced_groups = [
        group for group in groups if any(row.clip_id in forced_seed_clip_ids for row in group)
    ]
    forced_seed_count = sum(len(group) for group in forced_groups)
    seed_count = max(seed_count, forced_seed_count)
    if test_count is not None:
        dev_count = total - seed_count - test_count
    if seed_count + dev_count > total or dev_count < 0:
        raise ValueError(
            f"Infeasible split targets: seed={seed_count}, dev={dev_count}, total={total}"
        )
    targets = {"seed": seed_count, "dev": dev_count, "test": total - seed_count - dev_count}
    assigned: dict[str, list[ClipRecord]] = {name: [] for name in targets}
    class_counts: dict[str, Counter[str]] = {name: Counter() for name in targets}

    forced_group_ids = {id(group) for group in forced_groups}
    for group in forced_groups:
        assigned["seed"].extend(group)
        class_counts["seed"].update(row.label for row in group)

    for group in groups:
        if id(group) in forced_group_ids:
            continue
        group_labels = Counter(row.label for row in group)
        group_size = len(group)
        group_tie = random_tie[id(group)]
        group_label_names = tuple(group_labels)
        candidates = [name for name, target in targets.items() if target > 0]

        def score(
            name: str,
            group_size: int = group_size,
            group_label_names: tuple[str, ...] = group_label_names,
            group_tie: float = group_tie,
        ) -> tuple[float, float, float]:
            target = targets[name]
            after = len(assigned[name]) + group_size
            overflow = max(0, after - target)
            fill = after / target
            label_crowding = sum(class_counts[name][label] for label in group_label_names) / max(
                1, len(assigned[name])
            )
            return (float(overflow > 0), fill + 0.1 * label_crowding, group_tie)

        destination = min(candidates, key=score)
        assigned[destination].extend(group)
        class_counts[destination].update(row.label for row in group)

    output: list[ClipRecord] = []
    for split, rows in assigned.items():
        output.extend(
            ClipRecord(
                clip_id=row.clip_id,
                path=row.path,
                label=row.label,
                split=split,
                group_id=row.group_id,
                size_bytes=row.size_bytes,
            )
            for row in rows
        )
    return sorted(output, key=lambda row: (row.split, row.label, row.clip_id))


def write_manifest(records: Iterable[ClipRecord], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            row = record.to_dict()
            row["path"] = Path(os.path.relpath(record.path, path.parent)).as_posix()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: Path, split: str | None = None) -> Iterator[ClipRecord]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            clip_path = Path(row["path"])
            if not clip_path.is_absolute():
                clip_path = (path.parent / clip_path).resolve()
            record = ClipRecord(
                clip_id=row["clip_id"],
                path=clip_path,
                label=row["label"],
                split=row.get("split", "unassigned"),
                group_id=row.get("group_id"),
                size_bytes=int(row.get("size_bytes", 0)),
            )
            if split is None or record.split == split:
                yield record


def summarize(records: Iterable[ClipRecord]) -> dict[str, object]:
    rows = list(records)
    labels_by_split = {
        split: dict(sorted(Counter(row.label for row in rows if row.split == split).items()))
        for split in sorted({row.split for row in rows})
    }
    return {
        "clips": len(rows),
        "size_gib": round(sum(row.size_bytes for row in rows) / (1024**3), 3),
        "labels": dict(sorted(Counter(row.label for row in rows).items())),
        "splits": dict(sorted(Counter(row.split for row in rows).items())),
        "labels_by_split": labels_by_split,
        "groups": len({row.group_id for row in rows}),
        "unlabeled_files_excluded": "see inspect-data output",
    }
