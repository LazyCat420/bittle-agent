"""Observation / action / reward / termination spec — ONE source of truth.

Every function takes ``xp`` (``numpy`` or ``jax.numpy``) so the CPU evaluator
and the GPU trainer compute exactly the same quantities. Nothing here touches
a simulator: callers hand in physical quantities and get numbers back.

Observation (policy order joints, ``history_n`` = N):

    gravity (3) | gyro*0.25 (3) | cmd/scale (3) | (targets-stand)/60 (8N) | last action (8) | [phase sin,cos (2)]

Everything in it is available on the real robot from the IMU plus the
command history — no joint encoders, no base velocity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..assets.joint_map import POLICY_JOINTS, STAND_AGENT_DEG

REPO = Path(__file__).resolve().parent.parent.parent
PROFILE_PATH = REPO / "hardware_profiles" / "bittle-standard-biboard-v1-p1s.json"

N_JOINTS = 8
GYRO_SCALE = 0.25
CMD_SCALE = np.array([0.2, 0.1, 1.0])
TARGET_NORM_DEG = 60.0
STAND_DEG = np.array([STAND_AGENT_DEG[i] for i in POLICY_JOINTS], dtype=np.float64)
MAX_LATENCY = 6
FALL_UP_MIN = math.cos(1.0)  # |roll| or |pitch| > 1 rad
FALL_Z_MIN = 0.02
BODY_CONTACT_MAX_STEPS = 10  # 0.2 s at 50 Hz
ZERO_CMD_PROB = 0.1
ZERO_CMD_EPS = 0.02
#: Swing-foot height target above the local terrain (m); gaits clear 15-25 mm on flat ground.
FOOT_CLEARANCE_TARGET = 0.012
SWING_VEL_MIN = 0.02
#: A joint counts as stalled at >= 90% of its torque cap while barely moving.
STALL_TORQUE_FRACTION = 0.9
STALL_VEL_RAD_S = 0.1


#: Contact sensors whose firing means a non-foot body part is on the ground (fall termination).
#: Explicit, not a name filter: the shank/thigh "stumble" sensors also end in _floor_found and
#: must never join this set.
BODY_CONTACT_SENSORS = ("torso_floor_found", "head__1_floor_found",
                        "torso_col_floor_found", "head__1_col_floor_found", "jaw_1_col_floor_found")


def limb_contact_sensor_names() -> list[str]:
    """Shank + thigh ground-contact sensors (terrain XML variants only), policy-leg order."""
    from ..assets.joint_map import LEGS

    return [f"{leg}_{part}_floor_found" for leg in LEGS for part in ("shank", "thigh")]


def load_envelope(tier: str = "agent", path: Path = PROFILE_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Per-policy-joint [lo, hi] in bittle-agent degrees from the hardware profile."""
    prof = json.loads(path.read_text())
    lo = np.array([prof["envelopes"][str(i)][tier][0] for i in POLICY_JOINTS], dtype=np.float64)
    hi = np.array([prof["envelopes"][str(i)][tier][1] for i in POLICY_JOINTS], dtype=np.float64)
    return lo, hi


def obs_layout(history_n: int, phase_clock: bool) -> list[tuple[str, int]]:
    layout = [("gravity", 3), ("gyro", 3), ("command", 3),
              ("target_history", N_JOINTS * history_n), ("last_action", N_JOINTS)]
    if phase_clock:
        layout.append(("phase", 2))
    return layout


def obs_size(history_n: int, phase_clock: bool) -> int:
    return sum(n for _, n in obs_layout(history_n, phase_clock))


def build_obs(xp, gravity, gyro, cmd, target_hist_deg, last_action, phase=None):
    """``target_hist_deg``: (N, 8) newest first, in bittle-agent degrees."""
    parts = [
        gravity,
        gyro * GYRO_SCALE,
        cmd / xp.asarray(CMD_SCALE),
        ((target_hist_deg - xp.asarray(STAND_DEG)) / TARGET_NORM_DEG).reshape(-1),
        last_action,
    ]
    if phase is not None:
        parts.append(phase)
    return xp.concatenate(parts).astype(xp.float32)


def process_action(xp, action, prev_target_deg, scale_deg, lo, hi, mode: str):
    """Policy action in [-1,1]^8 -> new commanded target in integer degrees."""
    a = xp.clip(action, -1.0, 1.0)
    if mode == "residual_from_stand":
        target = xp.asarray(STAND_DEG) + a * (scale_deg * 5.0)
    else:
        target = prev_target_deg + a * scale_deg
    target = xp.clip(target, lo, hi)
    return xp.round(target)


SAT_MARGIN_DEG = 5.0


def target_saturation(xp, target_deg, lo, hi):
    """Per-joint edge proximity: 0 more than SAT_MARGIN_DEG inside the envelope, 1 at the edge.

    Measured in degrees from each edge (not normalised to the centre) because the
    envelopes are asymmetric around STAND — the rear shoulders stand 5 deg from
    their lower bound, which must not read as saturation.
    """
    d_lo = xp.maximum(lo + SAT_MARGIN_DEG - target_deg, 0.0)
    d_hi = xp.maximum(target_deg - (hi - SAT_MARGIN_DEG), 0.0)
    return xp.minimum((d_lo + d_hi) / SAT_MARGIN_DEG, 1.0)


def reward_terms(xp, q: dict[str, Any], tracking_sigma: float, ang_tracking_sigma: float,
                 base_height_target: float) -> dict[str, Any]:
    """Raw (unweighted) reward terms. Keys match ``RewardWeights`` fields.

    ``q`` keys: cmd(3), local_linvel(3), gyro(3), global_linvel(3), global_angvel(3),
    gravity(3), up_world(3), torso_z, terrain_h [= support height: mean terrain height under the
    four feet, so a stair edge is half a riser, not a step function], action(8), last_action(8), torques(8),
    joint_vel(8), torque_cap(8), target_norm(8) [0..1 saturation], target_deg(8),
    feet_air_time(4), first_contact(4), contact(4), feet_vel_xy(4,2), foot_clearance(4),
    limb_contact(8), uphill_xy(2).

    Terrain-aware terms are exact no-ops on flat ground: ``terrain_h`` is 0,
    ``uphill_xy`` is 0 and ``limb_contact`` never fires, so a flat run's reward is
    bit-identical to the pre-terrain trainer. ``orientation`` reads the torso
    up-vector in the WORLD frame (the terrain-normal frame under the tilted-world
    incline), which equals ``sum(gravity_body[:2]**2)`` on flat ground.
    """
    cmd = q["cmd"]
    cmd_norm = xp.linalg.norm(cmd)
    moving = cmd_norm > ZERO_CMD_EPS
    lin_err = xp.sum(xp.square(cmd[:2] - q["local_linvel"][:2]))
    ang_err = xp.square(cmd[2] - q["gyro"][2])
    swing = (1.0 - q["contact"]) * (xp.linalg.norm(q["feet_vel_xy"], axis=-1) > SWING_VEL_MIN)
    return {
        "tracking_lin_vel": xp.exp(-lin_err / tracking_sigma),
        "tracking_ang_vel": xp.exp(-ang_err / ang_tracking_sigma),
        "lin_vel_z": xp.square(q["global_linvel"][2]),
        "ang_vel_xy": xp.sum(xp.square(q["global_angvel"][:2])),
        "orientation": xp.sum(xp.square(q["up_world"][:2])),
        "base_height": xp.square((q["torso_z"] - q["terrain_h"]) - base_height_target) * 1000.0,  # mm^2/1000 -> ~O(1)
        "action_rate": xp.sum(xp.square(q["action"] - q["last_action"])),
        "energy": xp.sum(xp.abs(q["torques"] * q["joint_vel"])),
        "joint_saturation": xp.sum(q["target_norm"]),
        "feet_air_time": xp.sum((q["feet_air_time"] - 0.1) * q["first_contact"]) * moving,
        "feet_slip": xp.sum(xp.sum(xp.square(q["feet_vel_xy"]), axis=-1) * q["contact"]) * moving,
        "stand_still": xp.sum(xp.abs(q["target_deg"] - xp.asarray(STAND_DEG))) / TARGET_NORM_DEG * (~moving),
        # ── terrain / servo-safety terms (default weight 0.0; see TerrainConfig) ──
        # swing-foot height error vs FOOT_CLEARANCE_TARGET, only while airborne AND moving,
        # so a permanently high stance earns nothing
        # mm^2/1000 like base_height: unscaled (m^2) a 12 mm miss was 1.4e-4 per foot and the term stayed
        # < 0.2 % of the reward at the weight bound (house_v1 rungs h1/h2, 2026-09-08)
        # ONE-SIDED: only a swing BELOW the target is penalised (the gate wants a per-swing peak >= 8 mm; the
        # symmetric form also punished a 20 mm step, so the cheapest policy hovered at the target while dragging)
        "foot_clearance": xp.sum(xp.square(xp.maximum(FOOT_CLEARANCE_TARGET - q["foot_clearance"], 0.0)) * swing) * 1000.0,
        # shank / thigh touching the ground: the edge-collision ("stumble") penalty
        "stumble": xp.sum(q["limb_contact"]),
        # height gained per second up the slope; exactly 0 on flat ground
        "slope_progress": xp.maximum(xp.sum(q["global_linvel"][:2] * q["uphill_xy"]), 0.0) * moving,
        # joints pinned at the torque cap while not moving: a stalled servo (1.5 A each on the P1S)
        "stall": xp.sum(((q["torques"] >= STALL_TORQUE_FRACTION * q["torque_cap"]) & (xp.abs(q["joint_vel"]) < STALL_VEL_RAD_S)).astype(q["torques"].dtype)),
    }


def weighted_reward(xp, terms: dict[str, Any], weights: dict[str, float], dt: float):
    total = sum(terms[k] * float(weights[k]) for k in weights)
    return xp.clip(total * dt / 0.02, 0.0, 1e4)  # normalised so weights mean the same at 25 and 50 Hz


def termination(xp, up_z, torso_z, body_contact_steps, terrain_h=0.0):
    """Fallen: tilted past 1 rad, torso below FALL_Z_MIN above the local terrain, or a
    non-foot body part on the ground for BODY_CONTACT_MAX_STEPS."""
    return (up_z < FALL_UP_MIN) | ((torso_z - terrain_h) < FALL_Z_MIN) | (body_contact_steps >= BODY_CONTACT_MAX_STEPS)


def sample_command(rng_uniform, ranges: dict[str, tuple[float, float]]):
    """``rng_uniform(lo, hi, shape)`` -> command (vx, vy, wz); 10% of commands are zero."""
    vx = rng_uniform(ranges["vx"][0], ranges["vx"][1], ())
    vy = rng_uniform(ranges["vy"][0], ranges["vy"][1], ())
    wz = rng_uniform(ranges["wz"][0], ranges["wz"][1], ())
    zero = rng_uniform(0.0, 1.0, ()) < ZERO_CMD_PROB
    return vx, vy, wz, zero


def gravity_from_xmat(xp, xmat3x3):
    """Gravity direction in the body frame: R^T @ [0,0,-1]."""
    return -xmat3x3[2, :]
