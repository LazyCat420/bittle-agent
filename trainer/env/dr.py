"""Domain randomisation: sample parameters from ``DRConfig`` ranges.

Two layers, mirrored in both envs:

* **model-level** (per env in the GPU batch, per episode on CPU): ground
  friction, link masses (+payload with COM shift), servo kp, forcerange,
  joint damping / frictionloss, and the terrain field (gravity direction for
  the tilted-world incline, box placement for rocks/edges).
* **episode-level** (info): control latency, gyro bias, initial joint noise
  and tilt, push schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import terrain as tr


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
    #: per-env terrain: gravity (tilted world) + box placement; flat when absent
    terrain: tr.TerrainField = field(default_factory=tr.flat_field)


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
                       kp=10.0, forcerange=0.25, damping=0.05, frictionloss=0.01)


def nominal_episode_params() -> EpisodeParams:
    return EpisodeParams(latency_steps=1, gyro_bias=np.zeros(3), init_joint_noise_deg=np.zeros(8),
                         init_tilt_rad=np.zeros(2), push_enabled=False, push_vel=0.0, push_interval_s=5.0)


def sample_model_params(rng: np.random.Generator, dr, terrain_cfg=None) -> ModelParams:
    """Terrain is sampled whenever ``terrain_cfg`` is given, independently of ``dr.enabled``:
    the terrain is the task, DR is the robustness noise around it."""
    field_ = tr.sample_field_numpy(rng, terrain_cfg) if terrain_cfg is not None else tr.flat_field()
    if not dr.enabled:
        p = nominal_model_params()
        p.terrain = field_
        return p
    u = rng.uniform
    return ModelParams(
        friction=u(*dr.friction), mass_scale=u(*dr.mass_scale), payload_kg=u(*dr.payload_g) / 1000.0,
        com_shift_m=np.array([u(*dr.com_shift_mm) / 1000.0, u(-dr.com_shift_mm[1], dr.com_shift_mm[1]) / 2000.0, 0.0]),
        kp=u(*dr.kp), forcerange=u(*dr.forcerange), damping=u(*dr.damping), frictionloss=u(*dr.frictionloss),
        terrain=field_,
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


def apply_model_params_mujoco(model, p: ModelParams, ground_geom_ids: np.ndarray, torso_body_id: int,
                              base_mass: np.ndarray, base_ipos: np.ndarray, base_damping: np.ndarray,
                              base_frictionloss: np.ndarray, box_geom_ids: np.ndarray | None = None) -> None:
    """In-place edit of a mujoco.MjModel from its stored nominal arrays.

    ``ground_geom_ids`` = the floor plane plus any terrain boxes (friction applies to all of
    them); ``box_geom_ids`` = the parked terrain boxes to place from ``p.terrain`` (flat
    variants have none).
    """
    model.geom_friction[ground_geom_ids, 0] = p.friction
    model.opt.gravity[:] = p.terrain.gravity
    if box_geom_ids is not None and len(box_geom_ids):
        k = len(box_geom_ids)
        model.geom_pos[box_geom_ids] = p.terrain.box_pos[:k]
        model.geom_size[box_geom_ids] = p.terrain.box_half[:k]
        model.geom_quat[box_geom_ids] = tr.quat_from_yaw(np, p.terrain.box_yaw[:k])
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
