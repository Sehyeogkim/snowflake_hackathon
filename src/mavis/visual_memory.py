from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .fingerprints import VideoFingerprint, fingerprint_clip, fingerprint_many, hamming
from .models import ClipRecord


@dataclass(frozen=True)
class VisualMemoryEntry:
    experience_id: str
    clip_id: str
    label: str
    duration_s: float
    hashes: tuple[int, ...]


class VisualMemoryIndex:
    """Auditable OpenCV scene-signature index over EverOS-backed experiences."""

    def __init__(self, entries: list[VisualMemoryEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> VisualMemoryIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            VisualMemoryEntry(
                experience_id=str(row["experience_id"]),
                clip_id=str(row["clip_id"]),
                label=str(row["label"]),
                duration_s=float(row["duration_s"]),
                hashes=tuple(int(value) for value in row["hashes"]),
            )
            for row in payload["entries"]
        ]
        return cls(entries)

    def save(self, path: Path) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "mavis-visual-memory-v1",
            "signature": "three 64-bit dHashes at 20/50/80 percent plus duration",
            "entries": [asdict(entry) for entry in self.entries],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def references(
        self,
        query: VideoFingerprint,
        top_k: int = 15,
        allowed_experience_ids: set[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        scored = []
        for entry in self.entries:
            if (
                allowed_experience_ids is not None
                and entry.experience_id not in allowed_experience_ids
            ):
                continue
            mean_hamming = sum(
                hamming(left, right) for left, right in zip(query.hashes, entry.hashes)
            ) / len(entry.hashes)
            # Duration is a weak temporal-context signal, capped so a differently
            # trimmed but visually close event can still be retrieved.
            distance = mean_hamming + min(8.0, abs(query.duration_s - entry.duration_s))
            scored.append(
                (f"visual:{entry.experience_id}", entry.experience_id, 1.0 / (1.0 + distance))
            )
        scored.sort(key=lambda row: (-row[2], row[1]))
        return scored[:top_k]


def build_visual_memory_index(
    clips: list[ClipRecord],
    experience_ids_by_clip: dict[str, str],
    workers: int = 8,
    proxy_dir: Path | None = None,
) -> tuple[VisualMemoryIndex, list[dict[str, str]]]:
    selected = [clip for clip in clips if clip.clip_id in experience_ids_by_clip]
    fingerprint_clips = []
    for clip in selected:
        proxy = proxy_dir / f"{clip.clip_id}.mp4" if proxy_dir else None
        path = proxy if proxy and proxy.exists() else clip.path
        fingerprint_clips.append(
            ClipRecord(
                clip_id=clip.clip_id,
                path=path,
                label=clip.label,
                split=clip.split,
                group_id=clip.group_id,
                size_bytes=path.stat().st_size,
            )
        )
    fingerprints, failures = fingerprint_many(fingerprint_clips, workers=workers)
    entries = [
        VisualMemoryEntry(
            experience_id=experience_ids_by_clip[item.clip_id],
            clip_id=item.clip_id,
            label=item.label,
            duration_s=item.duration_s,
            hashes=item.hashes,
        )
        for item in fingerprints
    ]
    entries.sort(key=lambda entry: entry.experience_id)
    return VisualMemoryIndex(entries), failures


def fingerprint_query(clip: ClipRecord) -> VideoFingerprint:
    return fingerprint_clip(clip)
