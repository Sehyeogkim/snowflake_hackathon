from __future__ import annotations

import re
from pathlib import Path

CANONICAL_LABELS = (
    "safe_walkway",
    "safe_walkway_violation",
    "authorized_intervention",
    "unauthorized_intervention",
    "closed_panel_cover",
    "opened_panel_cover",
    "safe_carrying",
    "carrying_overload_with_forklift",
)

UNSAFE_LABELS = frozenset(
    {
        "safe_walkway_violation",
        "unauthorized_intervention",
        "opened_panel_cover",
        "carrying_overload_with_forklift",
    }
)

_ALIASES = {
    "safewalkway": "safe_walkway",
    "safewalkwayviolation": "safe_walkway_violation",
    "walkwayviolation": "safe_walkway_violation",
    "authorizedintervention": "authorized_intervention",
    "unauthorizedintervention": "unauthorized_intervention",
    "closedpanelcover": "closed_panel_cover",
    "openedpanelcover": "opened_panel_cover",
    "openpanelcover": "opened_panel_cover",
    "safecarrying": "safe_carrying",
    "carryingoverloadwithforklift": "carrying_overload_with_forklift",
    "forkliftoverload": "carrying_overload_with_forklift",
}


def normalize_label(value: str) -> str:
    """Normalize model/directory labels into the fixed eight-class taxonomy."""
    value = re.sub(r"^\s*\d+\s*[_\-\s]+", "", value)
    compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return _ALIASES.get(compact, re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_"))


def infer_label_from_path(path: Path) -> str | None:
    """Infer a label from any ancestor or filename, preferring the nearest match."""
    for part in [path.stem, *reversed(path.parts[:-1])]:
        label = normalize_label(part)
        if label in CANONICAL_LABELS:
            return label
    return None


def is_unsafe(label: str) -> bool:
    return normalize_label(label) in UNSAFE_LABELS
