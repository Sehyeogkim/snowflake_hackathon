"""Decision overlay for the live demo.

Draws what MAVIS decided and why directly onto the frame, because the argument
is easier to watch than to read: long stretches of SKIP costing nothing, a cheap
classify when the scene changes, and a strong call fired exactly when the scene
becomes ambiguous.

Writes to an MP4 by default. WSL and headless boxes have no display, so
``cv2.imshow`` is opt-in rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import Action, Step

_COLOURS = {
    Action.SKIP: (150, 150, 150),
    Action.CLASSIFY: (200, 180, 60),
    Action.CHEAP_VLM: (90, 200, 220),
    Action.STRONG_VLM: (70, 120, 250),
    Action.MULTIFRAME: (200, 90, 250),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(image: np.ndarray, step: Step, *, detected: bool, spent: float) -> np.ndarray:
    """Return a copy of ``image`` with the decision panel drawn on it."""
    out = image.copy()
    h, w = out.shape[:2]
    action = step.decision.action
    colour = _COLOURS[action]

    panel_h = 104
    panel = out[:panel_h].astype(np.float32) * 0.25
    out[:panel_h] = panel.astype(np.uint8)
    cv2.rectangle(out, (0, 0), (10, panel_h), colour, -1)

    mm, ss = divmod(step.timestamp_s, 60)
    cv2.putText(out, f"{int(mm):02d}:{ss:05.2f}", (24, 32), _FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(out, action.value, (150, 32), _FONT, 0.8, colour, 2)

    reason = step.decision.reason
    cv2.putText(out, _fit(reason, w - 40), (24, 58), _FONT, 0.45, (215, 215, 215), 1)

    hits = step.decision.memory_hits
    meta = (
        f"hazard {step.hazard_prob_after:.2f}   "
        f"gain {step.observed_ig:+.3f} bits   "
        f"memory {hits} recalled   "
        f"spent {spent:.5f}"
    )
    cv2.putText(out, meta, (24, 82), _FONT, 0.45, (180, 220, 180), 1)

    _draw_belief_bar(out, step.hazard_prob_after, w, h)
    if detected:
        _draw_alert(out, w, h)
    return out


def _fit(text: str, width_px: int) -> str:
    limit = max(int(width_px / 8), 20)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _draw_belief_bar(image: np.ndarray, p: float, w: int, h: int) -> None:
    x0, x1 = 24, w - 24
    y = h - 28
    cv2.rectangle(image, (x0, y), (x1, y + 12), (60, 60, 60), -1)
    filled = int((x1 - x0) * min(max(p, 0.0), 1.0))
    colour = (70, 120, 250) if p >= 0.6 else (90, 200, 120)
    cv2.rectangle(image, (x0, y), (x0 + filled, y + 12), colour, -1)
    cv2.putText(image, "hazard belief", (x0, y - 6), _FONT, 0.4, (200, 200, 200), 1)


def _draw_alert(image: np.ndarray, w: int, h: int) -> None:
    box_w, box_h = 420, 70
    x0, y0 = (w - box_w) // 2, h // 2 - box_h // 2
    cv2.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 220), -1)
    cv2.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 2)
    cv2.putText(
        image, "HAZARD DETECTED", (x0 + 46, y0 + 46), _FONT, 1.1, (255, 255, 255), 3
    )


class OverlayWriter:
    """Lazily-opened MP4 writer sized from the first frame it receives."""

    def __init__(self, path: str | Path, fps: float = 6.0, show: bool = False):
        self.path = Path(path)
        self.fps = fps
        self.show = show
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self._writer.write(image)
        if self.show:
            try:
                cv2.imshow("MAVIS", image)
                cv2.waitKey(1)
            except cv2.error:
                # Headless OpenCV, or no display attached. The MP4 is still
                # being written, so downgrade rather than abort the run.
                print("live window unavailable (headless OpenCV); writing MP4 only")
                self.show = False

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.show:
            cv2.destroyAllWindows()

    def __enter__(self) -> "OverlayWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def console_line(step: Step) -> str:
    """One-line trace for the terminal, mirroring the demo script in the README."""
    mm, ss = divmod(step.timestamp_s, 60)
    action = step.decision.action.value
    tag = "  <-- forced by recall floor" if step.decision.forced else ""
    return (
        f"{int(mm):02d}:{ss:05.2f}  {action:<11} "
        f"p={step.hazard_prob_after:.2f}  {step.decision.reason}{tag}"
    )
