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
    gravity(3), torso_z, action(8), last_action(8), torques(8), joint_vel(8),
    target_norm(8) [0..1 saturation], target_deg(8), feet_air_time(4), first_contact(4),
    contact(4), feet_vel_xy(4,2).
    """
    cmd = q["cmd"]
    cmd_norm = xp.linalg.norm(cmd)
    moving = cmd_norm > ZERO_CMD_EPS
    lin_err = xp.sum(xp.square(cmd[:2] - q["local_linvel"][:2]))
    ang_err = xp.square(cmd[2] - q["gyro"][2])
    return {
        "tracking_lin_vel": xp.exp(-lin_err / tracking_sigma),
        "tracking_ang_vel": xp.exp(-ang_err / ang_tracking_sigma),
        "lin_vel_z": xp.square(q["global_linvel"][2]),
        "ang_vel_xy": xp.sum(xp.square(q["global_angvel"][:2])),
        "orientation": xp.sum(xp.square(q["gravity"][:2])),
        "base_height": xp.square(q["torso_z"] - base_height_target) * 1000.0,  # mm^2/1000 -> ~O(1)
        "action_rate": xp.sum(xp.square(q["action"] - q["last_action"])),
        "energy": xp.sum(xp.abs(q["torques"] * q["joint_vel"])),
        "joint_saturation": xp.sum(q["target_norm"]),
        "feet_air_time": xp.sum((q["feet_air_time"] - 0.1) * q["first_contact"]) * moving,
        "feet_slip": xp.sum(xp.sum(xp.square(q["feet_vel_xy"]), axis=-1) * q["contact"]) * moving,
        "stand_still": xp.sum(xp.abs(q["target_deg"] - xp.asarray(STAND_DEG))) / TARGET_NORM_DEG * (~moving),
    }


def weighted_reward(xp, terms: dict[str, Any], weights: dict[str, float], dt: float):
    total = sum(terms[k] * float(weights[k]) for k in weights)
    return xp.clip(total * dt / 0.02, 0.0, 1e4)  # normalised so weights mean the same at 25 and 50 Hz


def termination(xp, up_z, torso_z, body_contact_steps):
    return (up_z < FALL_UP_MIN) | (torso_z < FALL_Z_MIN) | (body_contact_steps >= BODY_CONTACT_MAX_STEPS)


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
