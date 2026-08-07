from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .candidates import extract_candidates
from .config import Settings
from .models import ClipRecord


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[resolved] = digest.hexdigest()
    return cache[resolved]


def plan_seed_jobs(
    clips: list[ClipRecord],
    settings: Settings,
    work_dir: Path,
    output: Path,
    run_id: str = "seed-v1",
) -> dict[str, object]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    jobs = []
    unique_images: set[Path] = set()
    for clip in clips:
        candidates = extract_candidates(
            clip,
            work_dir.resolve() / "candidates",
            analysis_fps=settings.analysis_fps,
            scan_width=settings.scan_width,
            jpeg_quality=settings.jpeg_quality,
            include_crop=settings.crop_strong,
        )
        for action in settings.actions:
            images = candidates.for_action(action)
            unique_images.update(path.resolve() for path in images)
            image_hashes = [_sha256(path, hash_cache) for path in images]
            model = settings.cheap_model if action == "cheap_single" else settings.strong_model
            identity = json.dumps(
                {
                    "clip_id": clip.clip_id,
                    "action": action,
                    "model": model,
                    "prompt_version": settings.prompt_version,
                    "image_hashes": image_hashes,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            jobs.append(
                {
                    "job_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "run_id": run_id,
                    "clip_id": clip.clip_id,
                    "primary_clip_label": clip.label,
                    "split": clip.split,
                    "action": action,
                    "model": model,
                    "prompt_version": settings.prompt_version,
                    "image_paths": [
                        Path(os.path.relpath(path, output.parent)).as_posix() for path in images
                    ],
                    "image_sha256": image_hashes,
                    "query_tag": f"mavis|seed|{run_id}|{clip.clip_id}|{action}",
                    "state": "planned",
                    "query_id": None,
                    "billed_credits": None,
                }
            )
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "clips": len(clips),
        "jobs": len(jobs),
        "unique_images": len(unique_images),
        "image_references": sum(len(job["image_paths"]) for job in jobs),
        "ledger": str(output),
        "ledger_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
