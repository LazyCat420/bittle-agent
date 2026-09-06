"""Motion Primitive Library — per-limb clips and composition operators.

A motion primitive is a short keyframe sequence targeting a specific joint
group (head, single leg, torso, full body).  GLM composes primitives into
complete movesets using sequential, parallel, or blended composition.

Safety: composition rejects overlapping joint groups in parallel mode;
all outputs go through TrajectoryValidator before execution.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from ..joints import STAND_POSE

# ── Joint Group Definitions ─────────────────────────────────────────────

JOINT_GROUPS: dict[str, list[int]] = {
    "head":        [0],
    "front_left":  [8, 12],
    "front_right": [9, 13],
    "rear_left":   [11, 15],
    "rear_right":  [10, 14],
    "front_legs":  [8, 9, 12, 13],
    "rear_legs":   [10, 11, 14, 15],
    "torso":       [8, 9, 10, 11],
    "knees":       [12, 13, 14, 15],
    "all":         [0, 8, 9, 10, 11, 12, 13, 14, 15],
}


class CompositionError(Exception):
    """Raised when primitives cannot be composed safely."""
    pass


# ── Primitive Schema ────────────────────────────────────────────────────

def make_primitive(
    name: str,
    group: str,
    frames: list[dict[str, Any]],
    *,
    description: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create a well-formed primitive dict.

    Each frame in *frames* must have at minimum ``{"angles": {joint_idx: angle}}``.
    Optional per-frame keys: ``speed_deg_per_step`` (default 8), ``delay_ms`` (default 150).
    """
    if group not in JOINT_GROUPS:
        raise CompositionError(f"Unknown joint group {group!r}; choose from {list(JOINT_GROUPS)}")

    valid_joints = set(JOINT_GROUPS[group])
    for i, frame in enumerate(frames):
        angles = frame.get("angles", {})
        for j in angles:
            jint = int(j)
            if jint not in valid_joints:
                raise CompositionError(
                    f"Primitive {name!r} frame {i}: joint {jint} is not in group {group!r} "
                    f"(allowed: {sorted(valid_joints)})"
                )

    normalised_frames = []
    for frame in frames:
        nf = {
            "angles": {int(k): v for k, v in frame["angles"].items()},
            "speed_deg_per_step": int(frame.get("speed_deg_per_step", 8)),
            "delay_ms": int(frame.get("delay_ms", 150)),
        }
        normalised_frames.append(nf)

    return {
        "name": name,
        "group": group,
        "joints": sorted(valid_joints),
        "description": description,
        "tags": tags or [],
        "frames": normalised_frames,
    }


# ── Composition Operators ───────────────────────────────────────────────

def _expand_frame_to_full_body(frame: dict[str, Any]) -> dict[str, Any]:
    """Expand a partial-joint frame into a full 9-DOF frame using STAND_POSE defaults."""
    full_angles = dict(STAND_POSE)
    for j, a in frame["angles"].items():
        full_angles[int(j)] = a
    return {
        "angles": full_angles,
        "speed_deg_per_step": frame.get("speed_deg_per_step", 8),
        "delay_ms": frame.get("delay_ms", 150),
    }


def _get_touched_joints(primitives: list[dict[str, Any]]) -> dict[str, set[int]]:
    """Map each primitive name to the set of joint indices it actually touches."""
    result = {}
    for prim in primitives:
        touched: set[int] = set()
        for frame in prim["frames"]:
            touched.update(int(j) for j in frame["angles"])
        result[prim["name"]] = touched
    return result


def compose_sequential(
    primitives: list[dict[str, Any]],
    transition_blend_ms: int = 0,
) -> list[dict[str, Any]]:
    """Concatenate primitives end-to-end into a single frame list.

    If *transition_blend_ms* > 0, insert an interpolation frame between each pair
    of adjacent primitives to smooth the transition.
    """
    if not primitives:
        return []

    result_frames: list[dict[str, Any]] = []

    for idx, prim in enumerate(primitives):
        expanded = [_expand_frame_to_full_body(f) for f in prim["frames"]]

        if idx > 0 and transition_blend_ms > 0 and result_frames:
            # Interpolate between last frame of previous and first frame of current
            prev = result_frames[-1]
            curr = expanded[0]
            blend_angles = {}
            all_joints = set(prev["angles"]) | set(curr["angles"])
            for j in all_joints:
                a = prev["angles"].get(j, STAND_POSE.get(j, 0))
                b = curr["angles"].get(j, STAND_POSE.get(j, 0))
                blend_angles[j] = int(round((a + b) / 2.0))
            result_frames.append({
                "angles": blend_angles,
                "speed_deg_per_step": max(
                    prev.get("speed_deg_per_step", 8),
                    curr.get("speed_deg_per_step", 8),
                ),
                "delay_ms": transition_blend_ms,
            })

        result_frames.extend(expanded)

    return result_frames


def compose_parallel(
    primitives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge joint-disjoint primitives into simultaneous full-body frames.

    Raises CompositionError if any two primitives touch the same joint.
    """
    if not primitives:
        return []

    # Collision check
    touched = _get_touched_joints(primitives)
    names = list(touched.keys())
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            overlap = touched[names[i]] & touched[names[k]]
            if overlap:
                raise CompositionError(
                    f"Cannot compose in parallel: primitives {names[i]!r} and {names[k]!r} "
                    f"both touch joints {sorted(overlap)}. Use sequential mode or split the primitives."
                )

    # Find max frame count across primitives
    max_frames = max(len(p["frames"]) for p in primitives)

    result_frames: list[dict[str, Any]] = []
    for frame_idx in range(max_frames):
        merged_angles: dict[int, int] = dict(STAND_POSE)
        max_speed = 8
        max_delay = 0

        for prim in primitives:
            if frame_idx < len(prim["frames"]):
                f = prim["frames"][frame_idx]
            else:
                # Hold last frame
                f = prim["frames"][-1]

            for j, a in f["angles"].items():
                merged_angles[int(j)] = a
            max_speed = max(max_speed, f.get("speed_deg_per_step", 8))
            max_delay = max(max_delay, f.get("delay_ms", 150))

        result_frames.append({
            "angles": merged_angles,
            "speed_deg_per_step": max_speed,
            "delay_ms": max_delay,
        })

    return result_frames


def compose_blend(
    primitives: list[dict[str, Any]],
    blend_frames: int = 2,
    blend_delay_ms: int = 100,
) -> list[dict[str, Any]]:
    """Sequential composition with multi-frame interpolation between primitives.

    Inserts *blend_frames* linearly interpolated transition frames between each pair.
    """
    if not primitives:
        return []

    result_frames: list[dict[str, Any]] = []

    for idx, prim in enumerate(primitives):
        expanded = [_expand_frame_to_full_body(f) for f in prim["frames"]]

        if idx > 0 and blend_frames > 0 and result_frames:
            prev = result_frames[-1]
            curr = expanded[0]
            all_joints = set(prev["angles"]) | set(curr["angles"])

            for step in range(1, blend_frames + 1):
                t = step / (blend_frames + 1)
                interp_angles = {}
                for j in all_joints:
                    a = prev["angles"].get(j, STAND_POSE.get(j, 0))
                    b = curr["angles"].get(j, STAND_POSE.get(j, 0))
                    interp_angles[j] = int(round(a + (b - a) * t))
                result_frames.append({
                    "angles": interp_angles,
                    "speed_deg_per_step": max(
                        prev.get("speed_deg_per_step", 8),
                        curr.get("speed_deg_per_step", 8),
                    ),
                    "delay_ms": blend_delay_ms,
                })

        result_frames.extend(expanded)

    return result_frames


def compose(
    primitives: list[dict[str, Any]],
    mode: str = "sequential",
    *,
    transition_blend_ms: int = 100,
    blend_frames: int = 2,
    blend_delay_ms: int = 100,
) -> list[dict[str, Any]]:
    """Compose primitives using the specified mode.

    Modes:
        sequential — end-to-end concatenation with optional transition smoothing
        parallel   — merge into simultaneous frames (rejects joint collisions)
        blend      — sequential with multi-frame linear interpolation between primitives
    """
    if mode == "sequential":
        return compose_sequential(primitives, transition_blend_ms=transition_blend_ms)
    elif mode == "parallel":
        return compose_parallel(primitives)
    elif mode == "blend":
        return compose_blend(primitives, blend_frames=blend_frames, blend_delay_ms=blend_delay_ms)
    else:
        raise CompositionError(f"Unknown composition mode {mode!r}; use 'sequential', 'parallel', or 'blend'")


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute quality metrics for a composed frame sequence.

    Returns smoothness score, duration estimate, safety margins, and reversal count.
    """
    if not frames:
        return {"smoothness": 0, "total_duration_ms": 0, "max_delta": 0, "reversal_count": 0, "frame_count": 0}

    total_delta = 0.0
    max_delta = 0.0
    delta_count = 0
    reversals: dict[int, int] = {}
    last_deltas: dict[int, float] = {}
    total_duration = 0
    prev_angles: dict[int, int] | None = None

    for frame in frames:
        angles = {int(j): a for j, a in frame["angles"].items()}
        total_duration += frame.get("delay_ms", 150)

        if prev_angles is not None:
            for j in angles:
                if j in prev_angles:
                    d = angles[j] - prev_angles[j]
                    ad = abs(d)
                    total_delta += ad
                    delta_count += 1
                    if ad > max_delta:
                        max_delta = ad

                    if ad > 2.0:
                        last_d = last_deltas.get(j, 0.0)
                        if (d > 0 and last_d < -2.0) or (d < 0 and last_d > 2.0):
                            reversals[j] = reversals.get(j, 0) + 1
                        last_deltas[j] = d

        prev_angles = angles

    avg_delta = total_delta / max(1, delta_count)
    # Smoothness: inverse of average delta, scaled 0-100 (lower delta = smoother)
    smoothness = max(0, min(100, int(100 - avg_delta * 2)))
    total_reversals = sum(reversals.values())

    return {
        "smoothness": smoothness,
        "avg_delta_deg": round(avg_delta, 1),
        "max_delta_deg": round(max_delta, 1),
        "total_duration_ms": total_duration,
        "frame_count": len(frames),
        "reversal_count": total_reversals,
        "reversals_by_joint": reversals,
    }
