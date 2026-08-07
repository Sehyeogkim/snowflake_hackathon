"""Minimal .env loader.

A dependency for reading `KEY=value` lines is not worth taking. Existing
environment variables always win, so `GEMINI_API_KEY=... mavis bench` overrides
the file without editing it.
"""

from __future__ import annotations

import os
from pathlib import Path


def load(path: str | Path = ".env", *, override: bool = False) -> int:
    """Load ``path`` into ``os.environ``. Returns how many keys were set."""
    p = Path(path)
    if not p.exists():
        return 0
    count = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if not key or (not override and key in os.environ):
            continue
        os.environ[key] = value
        count += 1
    return count
