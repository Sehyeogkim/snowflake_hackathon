"""The Mendeley 'Safe and Unsafe Behaviours' dataset (xjmtb22pff).

Clip-level class labels are our hazard ground truth: four unsafe classes, each
paired with a visually similar safe counterpart. The pairing is the point — a
cheap classifier cannot separate ``0_safe_walkway_violation`` from
``4_safe_walkway`` reliably, which is exactly where a strong VLM call earns its
cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: class id -> (folder name, hazard?, human description, paired counterpart id)
CLASSES: dict[int, tuple[str, bool, str, int]] = {
    0: ("0_safe_walkway_violation", True, "worker outside the designated walkway", 4),
    1: ("1_unauthorized_intervention", True, "unauthorised intervention on machinery", 5),
    2: ("2_opened_panel_cover", True, "electrical panel cover left open", 6),
    3: ("3_carrying_overload_with_forklift", True, "forklift carrying an unsafe load", 7),
    4: ("4_safe_walkway", False, "worker inside the designated walkway", 0),
    5: ("5_authorized_intervention", False, "authorised intervention on machinery", 1),
    6: ("6_closed_panel_cover", False, "electrical panel cover closed", 2),
    7: ("7_safe_carrying", False, "forklift carrying a normal load", 3),
}

HAZARD_CLASS_IDS = [i for i, m in CLASSES.items() if m[1]]
SAFE_CLASS_IDS = [i for i, m in CLASSES.items() if not m[1]]

#: Category strings handed to AI_CLASSIFY as the cheap first look.
CLASSIFY_CATEGORIES = [
    "worker walking outside marked walkway",
    "worker walking inside marked walkway",
    "person reaching into or servicing running machinery",
    "technician servicing machinery with guards and permits in place",
    "electrical panel with its cover open",
    "electrical panel with its cover closed",
    "forklift carrying an oversized or unstable load",
    "forklift carrying a normal secured load",
    "no person or vehicle of interest in view",
]


@dataclass(slots=True)
class Clip:
    """One labelled video file on disk."""

    path: Path
    class_id: int
    split: str

    @property
    def clip_id(self) -> str:
        return f"{self.split}/{self.class_folder}/{self.path.stem}"

    @property
    def class_folder(self) -> str:
        return CLASSES[self.class_id][0]

    @property
    def is_hazard(self) -> bool:
        return CLASSES[self.class_id][1]

    @property
    def description(self) -> str:
        return CLASSES[self.class_id][2]

    @property
    def counterpart_id(self) -> int:
        """The visually similar clip class with the opposite safety label."""
        return CLASSES[self.class_id][3]


def _class_id_from_folder(name: str) -> int | None:
    normalised = name.replace(" ", "_")
    for cid, (folder, *_rest) in CLASSES.items():
        if folder == normalised:
            return cid
    return None


def discover(root: str | Path, split: str | None = None) -> list[Clip]:
    """Find labelled clips under ``root`` (expects ``<root>/<split>/<class>/*.mp4``).

    Tolerates a partially downloaded tree — clips still in flight simply do not
    appear yet, so the pipeline is runnable before the fetch finishes.
    """
    root = Path(root)
    clips: list[Clip] = []
    if not root.exists():
        return clips
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if split and split_dir.name != split:
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cid = _class_id_from_folder(class_dir.name)
            if cid is None:
                continue
            for video in sorted(class_dir.glob("*.mp4")):
                clips.append(Clip(path=video, class_id=cid, split=split_dir.name))
    return clips


def readable(clips: list[Clip]) -> tuple[list[Clip], list[Clip]]:
    """Split clips into ``(readable, unreadable)``.

    A download interrupted mid-write leaves a truncated MP4 that OpenCV cannot
    open. Those must be reported, never silently dropped: a benchmark that
    quietly ran on fewer clips than it claimed is worse than one that failed.
    """
    import cv2

    ok, bad = [], []
    for clip in clips:
        cap = cv2.VideoCapture(str(clip.path))
        (ok if cap.isOpened() else bad).append(clip)
        cap.release()
    return ok, bad


def summarise(clips: list[Clip]) -> str:
    """One-line-per-class inventory, used by ``mavis data``."""
    from collections import Counter

    counts = Counter((c.split, c.class_id) for c in clips)
    lines = [f"{'split/class':48s} {'n':>3s}  hazard"]
    for (split, cid), n in sorted(counts.items()):
        folder, hazard, _desc, _pair = CLASSES[cid]
        lines.append(f"{split + '/' + folder:48s} {n:3d}  {'YES' if hazard else 'no'}")
    lines.append(f"{'TOTAL':48s} {len(clips):3d}")
    return "\n".join(lines)
