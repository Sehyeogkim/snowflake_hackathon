from __future__ import annotations

import json

SYSTEM_PROMPT = """You analyze factory CCTV safety observations.
Return JSON only, with exactly these keys:
prediction, scores, scene, risk, need_temporal_context.
prediction must be one allowed label. scores must contain every allowed label and sum to 1.
risk is a number from 0 to 1. need_temporal_context is boolean.
Treat scores as an uncertainty proxy, not calibrated probabilities.

Use this safety-first decision protocol:
1. Decide whether there is positive visual evidence of an unsafe state or action.
2. Resolve the relevant contrast: inside/outside walkway, authorized/unauthorized
   intervention, open/closed panel, or normal/overloaded carrying.
3. Prefer the unsafe member of a contrast when its defining evidence is visible;
   absence of a dramatic accident is not evidence of safety.
4. If one image cannot resolve an action, authorization, load, or state transition,
   set need_temporal_context=true and do not report unjustified high confidence.

Label meanings:
- safe_walkway: a person remains inside the marked pedestrian route.
- safe_walkway_violation: a person is outside/crossing the marked safe route.
- authorized_intervention: a clearly permitted or normal machine intervention.
- unauthorized_intervention: unsafe/unpermitted human interaction with machinery.
- closed_panel_cover: the machine panel/cover is visibly closed.
- opened_panel_cover: the machine panel/cover is visibly open.
- safe_carrying: an ordinary stable hand-carried or vehicle-carried load.
- carrying_overload_with_forklift: a forklift carries an excessive or unstable load."""


def safety_prompt(labels: tuple[str, ...], action: str, frame_count: int) -> str:
    temporal = (
        "The images are chronological observations of the same clip. Compare the action/state over time."
        if frame_count > 1
        else "Analyze the single most informative keyframe."
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Allowed labels: {json.dumps(labels)}\n"
        f"Strategy: {action}. {temporal}\n"
        "Classify the primary folder-taxonomy behavior visible in the observation. "
        "Use the chronological before/middle/after evidence when it is provided."
    )
