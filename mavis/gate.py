"""OpenCV cheap gate — the free filter in front of every paid inference.

Nothing here calls an AI model. It only answers "is this frame visually the same
as the last one I let through?", which removes long static stretches of CCTV
before they ever cost a token. Both the baseline and MAVIS run behind the same
gate, so the benchmark measures scheduling, not preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import GateConfig
from .types import Frame


def dhash(image: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash: robust to compression noise, sensitive to content."""
    small = cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass(slots=True)
class GateStats:
    decoded: int = 0
    passed: int = 0
    dropped_duplicate: int = 0
    dropped_no_motion: int = 0
    forced_by_gap: int = 0

    @property
    def drop_rate(self) -> float:
        return 0.0 if not self.decoded else 1.0 - self.passed / self.decoded


class CheapGate:
    """Stateful duplicate/no-motion filter over a stream of frames."""

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.stats = GateStats()
        self._prev_gray: np.ndarray | None = None
        self._prev_hash: int | None = None
        self._last_pass_ts: float | None = None

    def _work_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.cfg.work_width / max(w, 1)
        small = cv2.resize(image, (self.cfg.work_width, max(int(h * scale), 1)))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def accept(self, image: np.ndarray, timestamp_s: float) -> bool:
        """Decide whether this frame is worth showing to anything downstream."""
        self.stats.decoded += 1
        gray = self._work_image(image)
        h = dhash(gray)

        if self._prev_gray is None:
            self._admit(gray, h, timestamp_s)
            return True

        gap = timestamp_s - (self._last_pass_ts or 0.0)
        if gap >= self.cfg.max_gap_s:
            self.stats.forced_by_gap += 1
            self._admit(gray, h, timestamp_s)
            return True

        if self._prev_hash is not None and hamming(h, self._prev_hash) < self.cfg.dhash_threshold:
            self.stats.dropped_duplicate += 1
            return False

        motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
        if motion < self.cfg.motion_threshold:
            self.stats.dropped_no_motion += 1
            return False

        self._admit(gray, h, timestamp_s)
        return True

    def _admit(self, gray: np.ndarray, h: int, timestamp_s: float) -> None:
        self._prev_gray = gray
        self._prev_hash = h
        self._last_pass_ts = timestamp_s
        self.stats.passed += 1


def iter_gated_frames(
    video_path: str | Path, cfg: GateConfig
) -> Iterator[tuple[Frame, GateStats]]:
    """Decode ``video_path`` and yield only the frames that clear the gate.

    Yields ``(frame, stats)`` so a caller can report the drop rate without
    reaching into the gate. ``stats`` is the live object, not a copy.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    gate = CheapGate(cfg)
    index = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % cfg.decode_stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                ts = index / fps
                if gate.accept(image, ts):
                    yield Frame(index=index, timestamp_s=ts, image=image), gate.stats
            index += 1
    finally:
        cap.release()


def probe(video_path: str | Path) -> tuple[float, float, int]:
    """Return ``(duration_s, fps, frame_count)`` without decoding pixels."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return (count / fps if fps else 0.0), fps, count
    finally:
        cap.release()
