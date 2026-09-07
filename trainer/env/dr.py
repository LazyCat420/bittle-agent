"""Domain randomisation: sample parameters from ``DRConfig`` ranges.

Two layers, mirrored in both envs:

* **model-level** (per env in the GPU batch, per episode on CPU): floor
  friction, link masses (+payload with COM shift), servo kp, forcerange,
  joint damping / frictionloss.
* **episode-level** (info): control latency, gyro bias, initial joint noise
  and tilt, push schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelParams:
    friction: float
    mass_scale: float
    payload_kg: float
    com_shift_m: np.ndarray  # (3,)
    kp: float
    forcerange: float
    damping: float
    frictionloss: float


@dataclass
class EpisodeParams:
    latency_steps: int
    gyro_bias: np.ndarray  # (3,)
    init_joint_noise_deg: np.ndarray  # (8,)
    init_tilt_rad: np.ndarray  # (2,) roll, pitch
    push_enabled: bool
    push_vel: float
    push_interval_s: float


def nominal_model_params() -> ModelParams:
    return ModelParams(friction=0.9, mass_scale=1.0, payload_kg=0.0, com_shift_m=np.zeros(3),
                       kp=40.0, forcerange=0.2, damping=1.5, frictionloss=0.15)


def nominal_episode_params() -> EpisodeParams:
    return EpisodeParams(latency_steps=1, gyro_bias=np.zeros(3), init_joint_noise_deg=np.zeros(8),
                         init_tilt_rad=np.zeros(2), push_enabled=False, push_vel=0.0, push_interval_s=5.0)


def sample_model_params(rng: np.random.Generator, dr) -> ModelParams:
    if not dr.enabled:
        return nominal_model_params()
    u = rng.uniform
    return ModelParams(
        friction=u(*dr.friction), mass_scale=u(*dr.mass_scale), payload_kg=u(*dr.payload_g) / 1000.0,
        com_shift_m=np.array([u(*dr.com_shift_mm) / 1000.0, u(-dr.com_shift_mm[1], dr.com_shift_mm[1]) / 2000.0, 0.0]),
        kp=u(*dr.kp), forcerange=u(*dr.forcerange), damping=u(*dr.damping), frictionloss=u(*dr.frictionloss),
    )


def sample_episode_params(rng: np.random.Generator, dr) -> EpisodeParams:
    if not dr.enabled:
        p = nominal_episode_params()
        p.push_enabled = bool(dr.push_enabled)
        return p
    u = rng.uniform
    return EpisodeParams(
        latency_steps=int(rng.integers(dr.latency_steps[0], dr.latency_steps[1] + 1)),
        gyro_bias=u(-dr.gyro_bias, dr.gyro_bias, 3),
        init_joint_noise_deg=u(-dr.init_joint_noise_deg, dr.init_joint_noise_deg, 8),
        init_tilt_rad=u(-dr.init_tilt_rad, dr.init_tilt_rad, 2),
        push_enabled=bool(dr.push_enabled), push_vel=u(*dr.push_vel), push_interval_s=u(*dr.push_interval_s),
    )


def apply_model_params_mujoco(model, base, p: ModelParams, floor_geom_id: int, torso_body_id: int,
                              base_mass: np.ndarray, base_ipos: np.ndarray, base_damping: np.ndarray,
                              base_frictionloss: np.ndarray) -> None:
    """In-place edit of a mujoco.MjModel from its stored nominal arrays."""
    model.geom_friction[floor_geom_id, 0] = p.friction
    model.body_mass[:] = base_mass * p.mass_scale
    model.body_mass[torso_body_id] += p.payload_kg
    model.body_ipos[:] = base_ipos
    model.body_ipos[torso_body_id] = base_ipos[torso_body_id] + p.com_shift_m
    model.actuator_gainprm[:, 0] = p.kp
    model.actuator_biasprm[:, 1] = -p.kp
    model.actuator_forcerange[:, 0] = -p.forcerange
    model.actuator_forcerange[:, 1] = p.forcerange
    model.dof_damping[6:] = p.damping
    model.dof_frictionloss[6:] = p.frictionloss
