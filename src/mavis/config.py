from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    labels: tuple[str, ...]
    unsafe_labels: frozenset[str]
    cheap_model: str
    strong_model: str
    analysis_fps: float
    scan_width: int
    jpeg_quality: int
    max_seed_clips: int
    max_workers: int
    actions: tuple[str, ...]
    crop_strong: bool
    prompt_version: str
    safety_recall_tolerance_pp: float

    @classmethod
    def load(cls, path: Path) -> Settings:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            labels=tuple(data["labels"]),
            unsafe_labels=frozenset(data["unsafe_labels"]),
            cheap_model=str(data["cheap_model"]),
            strong_model=str(data["strong_model"]),
            analysis_fps=float(data["analysis_fps"]),
            scan_width=int(data["scan_width"]),
            jpeg_quality=int(data["jpeg_quality"]),
            max_seed_clips=int(data["max_seed_clips"]),
            max_workers=int(data["max_workers"]),
            actions=tuple(data["actions"]),
            crop_strong=bool(data["crop_strong"]),
            prompt_version=str(data["prompt_version"]),
            safety_recall_tolerance_pp=float(data["safety_recall_tolerance_pp"]),
        )


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load a small .env file without overriding exported environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_env(*names: str) -> dict[str, str]:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}
