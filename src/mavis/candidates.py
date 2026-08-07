from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .models import CandidateFrame, CandidateSet, ClipRecord


class CandidateExtractionError(RuntimeError):
    pass


def _cache_signature(
    clip: ClipRecord,
    analysis_fps: float,
    scan_width: int,
    jpeg_quality: int,
    include_crop: bool,
) -> dict[str, object]:
    stat = clip.path.stat()
    return {
        "source_path": str(clip.path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "analysis_fps": analysis_fps,
        "scan_width": scan_width,
        "jpeg_quality": jpeg_quality,
        "include_crop": include_crop,
        "extractor_version": 3,
    }


def _load_cached(path: Path, signature: dict[str, object]) -> CandidateSet | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            return None

        def frame(name: str) -> CandidateFrame:
            row = payload[name]
            candidate = CandidateFrame(
                name=row["name"],
                frame_index=int(row["frame_index"]),
                timestamp_s=float(row["timestamp_s"]),
                path=Path(row["path"]),
                motion_score=float(row["motion_score"]),
                sharpness=float(row["sharpness"]),
            )
            if not candidate.path.exists():
                raise FileNotFoundError(candidate.path)
            return candidate

        crop = frame("crop") if payload.get("crop") else None
        return CandidateSet(
            clip_id=payload["clip_id"],
            fps=float(payload["fps"]),
            frame_count=int(payload["frame_count"]),
            duration_s=float(payload["duration_s"]),
            early=frame("early"),
            peak=frame("peak"),
            late=frame("late"),
            crop=crop,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _gray_thumb(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (width, height))


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _save_frame(frame: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise CandidateExtractionError(f"Could not write candidate frame: {path}")


def extract_candidates(
    clip: ClipRecord,
    output_root: Path,
    analysis_fps: float = 2.0,
    scan_width: int = 192,
    jpeg_quality: int = 85,
    include_crop: bool = False,
) -> CandidateSet:
    """Scan low-resolution samples once and decode only the selected full frames."""
    cache_path = output_root / clip.clip_id / "candidates.json"
    signature = _cache_signature(clip, analysis_fps, scan_width, jpeg_quality, include_crop)
    cached = _load_cached(cache_path, signature)
    if cached is not None:
        return cached
    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise CandidateExtractionError(f"Could not open video: {clip.path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise CandidateExtractionError(f"Video reports no frames: {clip.path}")
        duration_s = frame_count / fps
        step = max(1, round(fps / max(0.1, analysis_fps)))
        anchors = {
            max(0, round(frame_count * 0.15)),
            max(0, round(frame_count * 0.50)),
            min(frame_count - 1, round(frame_count * 0.85)),
        }
        sample_indices = sorted(set(range(0, frame_count, step)) | anchors | {frame_count - 1})
        sample_set = set(sample_indices)
        previous: np.ndarray | None = None
        scored: list[tuple[int, float, float]] = []
        encoded: dict[int, bytes] = {}
        for frame_index in range(frame_count):
            if not capture.grab():
                break
            if frame_index not in sample_set:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                continue
            gray = _gray_thumb(frame, scan_width)
            motion = 0.0 if previous is None else float(cv2.absdiff(gray, previous).mean() / 255.0)
            sharpness = _sharpness(gray)
            scored.append((frame_index, motion, sharpness))
            encode_ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not encode_ok:
                raise CandidateExtractionError(f"Could not encode frame {frame_index}: {clip.path}")
            encoded[frame_index] = buffer.tobytes()
            previous = gray
        if len(scored) < 3:
            raise CandidateExtractionError(
                f"Need at least three distinct sampled frames: {clip.path}"
            )

        # Motion is primary; log-scaled sharpness breaks blurry-frame ties.
        peak_position = max(
            range(len(scored)),
            key=lambda index: scored[index][1] + 0.02 * math.log1p(scored[index][2]),
        )
        # Adjacent 2 Hz samples are often three nearly identical frames. Require
        # a meaningful before/after gap so multi-frame reasoning observes state
        # change, while retaining sharp/moving evidence on either side.
        context_gap_frames = max(step, round(frame_count * 0.12))
        peak_frame = scored[peak_position][0]
        before = [
            index
            for index, item in enumerate(scored)
            if item[0] <= peak_frame - context_gap_frames
        ]
        after = [
            index
            for index, item in enumerate(scored)
            if item[0] >= peak_frame + context_gap_frames
        ]

        def evidence_score(index: int) -> float:
            _, motion, sharpness = scored[index]
            return motion + 0.02 * math.log1p(sharpness)

        early_position = max(before, key=evidence_score) if before else 0
        late_position = max(after, key=evidence_score) if after else len(scored) - 1
        chosen = [early_position, peak_position, late_position]
        if len(set(chosen)) < 3:
            chosen = [0, len(scored) // 2, len(scored) - 1]
        chosen.sort(key=lambda index: scored[index][0])
        desired = {
            "early": scored[chosen[0]],
            "peak": scored[chosen[1]],
            "late": scored[chosen[2]],
        }

        clip_dir = output_root / clip.clip_id
        candidates: dict[str, CandidateFrame] = {}
        for name, (frame_index, motion, sharpness) in desired.items():
            path = clip_dir / f"{name}_{frame_index:08d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded[frame_index])
            candidates[name] = CandidateFrame(
                name=name,
                frame_index=frame_index,
                timestamp_s=frame_index / fps,
                path=path.resolve(),
                motion_score=motion,
                sharpness=sharpness,
            )

        crop_candidate = None
        if include_crop:
            peak = cv2.imdecode(
                np.frombuffer(encoded[desired["peak"][0]], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            height, width = peak.shape[:2]
            x1, x2 = round(width * 0.2), round(width * 0.8)
            y1, y2 = round(height * 0.15), round(height * 0.9)
            crop_path = clip_dir / f"crop_{desired['peak'][0]:08d}.jpg"
            _save_frame(peak[y1:y2, x1:x2], crop_path, jpeg_quality)
            crop_candidate = CandidateFrame(
                name="crop",
                frame_index=desired["peak"][0],
                timestamp_s=desired["peak"][0] / fps,
                path=crop_path.resolve(),
                motion_score=desired["peak"][1],
                sharpness=desired["peak"][2],
            )

        result = CandidateSet(
            clip_id=clip.clip_id,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            early=candidates["early"],
            peak=candidates["peak"],
            late=candidates["late"],
            crop=crop_candidate,
        )
        payload = {
            "signature": signature,
            "clip_id": result.clip_id,
            "fps": result.fps,
            "frame_count": result.frame_count,
            "duration_s": result.duration_s,
            "early": result.early.to_dict(),
            "peak": result.peak.to_dict(),
            "late": result.late.to_dict(),
            "crop": result.crop.to_dict() if result.crop else None,
        }
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        return result
    finally:
        capture.release()
