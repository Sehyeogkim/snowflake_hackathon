from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2

from .models import ClipRecord


@dataclass(frozen=True)
class VideoFingerprint:
    clip_id: str
    label: str
    split: str
    duration_s: float
    hashes: tuple[int, ...]


def difference_hash(frame) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def fingerprint_clip(clip: ClipRecord) -> VideoFingerprint:
    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {clip.path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        hashes = []
        for ratio in (0.2, 0.5, 0.8):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, max(0, round(count * ratio))))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode fingerprint frame from {clip.path}")
            hashes.append(difference_hash(frame))
        return VideoFingerprint(
            clip_id=clip.clip_id,
            label=clip.label,
            split=clip.split,
            duration_s=count / fps,
            hashes=tuple(hashes),
        )
    finally:
        capture.release()


def fingerprint_many(
    clips: list[ClipRecord], workers: int = 8
) -> tuple[list[VideoFingerprint], list[dict[str, str]]]:
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fingerprint_clip, clip): clip for clip in clips}
        for future in as_completed(futures):
            clip = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {"clip_id": clip.clip_id, "path": str(clip.path), "error": str(exc)}
                )
    return results, failures


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_duplicate_pairs(
    fingerprints: list[VideoFingerprint],
    max_mean_hamming: float = 5.0,
    max_duration_delta_s: float = 0.75,
) -> list[dict[str, object]]:
    pairs = []
    by_label: dict[str, list[VideoFingerprint]] = {}
    for item in fingerprints:
        by_label.setdefault(item.label, []).append(item)
    for label, items in by_label.items():
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                if abs(left.duration_s - right.duration_s) > max_duration_delta_s:
                    continue
                distances = [hamming(a, b) for a, b in zip(left.hashes, right.hashes)]
                mean_distance = sum(distances) / len(distances)
                if mean_distance <= max_mean_hamming:
                    pairs.append(
                        {
                            "label": label,
                            "left_clip_id": left.clip_id,
                            "left_split": left.split,
                            "right_clip_id": right.clip_id,
                            "right_split": right.split,
                            "mean_hamming": mean_distance,
                            "duration_delta_s": abs(left.duration_s - right.duration_s),
                            "cross_split": left.split != right.split,
                        }
                    )
    return sorted(pairs, key=lambda row: (row["mean_hamming"], row["duration_delta_s"]))
