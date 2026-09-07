"""CPU MuJoCo environment (full-fidelity mesh model by default).

Same observation / action / reward spec as the GPU env (``spec.py``), driven
step by step with numpy. Used by the evaluator, the benchmark gates and the
rollout recorder; also usable for CPU-only training with subprocess workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..assets import joint_map as jm
from ..config import TrainConfig
from . import dr as dr_mod
from . import spec

GENERATED = Path(__file__).resolve().parent.parent / "assets" / "generated"


@dataclass
class StepInfo:
    terms: dict[str, float] = field(default_factory=dict)
    contact: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    target_deg: np.ndarray = field(default_factory=lambda: np.zeros(8))
    applied_deg: np.ndarray = field(default_factory=lambda: np.zeros(8))
    fell: bool = False


class BittleCpuEnv:
    """Single-instance MuJoCo env. Deterministic for a given seed."""

    def __init__(self, config: TrainConfig, *, variant: str = "cpu", seed: int = 0,
                 dr_override: dr_mod.ModelParams | None = None,
                 episode_override: dr_mod.EpisodeParams | None = None):
        self.cfg = config.resolved()
        self.variant = variant
        self.xml_path = GENERATED / f"bittle_{variant}.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.dr_override = dr_override
        self.episode_override = episode_override

        m = self.model
        self.n_substeps = int(round(self.cfg.control_dt / m.opt.timestep))
        self.qpos_idx = jm.qpos_indices(m)
        self.dof_idx = jm.dof_indices(m)
        self.torso_id = m.body("torso").id
        self.floor_id = m.geom("floor").id
        self.imu_site = m.site("imu_site").id
        self.lo, self.hi = spec.load_envelope("agent")
        self._sensor = {m.sensor(i).name: (m.sensor_adr[i], m.sensor_dim[i]) for i in range(m.nsensor)}
        self.foot_found = [f"{leg}_foot_floor_found" for leg in jm.LEGS]
        self.body_found = [n for n in self._sensor if n.endswith("_floor_found") and "foot" not in n]
        self.foot_vel = [f"{leg}_foot_global_linvel" for leg in jm.LEGS]
        self.foot_pos = [f"{leg}_foot_pos" for leg in jm.LEGS]
        # nominal arrays for DR
        self._base_mass = m.body_mass.copy()
        self._base_ipos = m.body_ipos.copy()
        self._base_damping = m.dof_damping.copy()
        self._base_frictionloss = m.dof_frictionloss.copy()
        self.key_qpos = m.key_qpos[0].copy()
        self.key_ctrl = m.key_ctrl[0].copy()
        self.stand_deg = spec.STAND_DEG.copy()
        self.history_n = self.cfg.obs.history_n
        self.reset_count = 0

    # ── helpers ────────────────────────────────────────────────────────
    def sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr:adr + dim]

    def _cmd_ranges(self) -> dict[str, tuple[float, float]]:
        c = self.cfg.commands
        return {"vx": c.vx, "vy": c.vy, "wz": c.wz}

    def _sample_command(self) -> np.ndarray:
        vx, vy, wz, zero = spec.sample_command(lambda lo, hi, shape: self.rng.uniform(lo, hi, shape), self._cmd_ranges())
        cmd = np.array([vx, vy, wz], dtype=np.float64)
        return np.zeros(3) if zero else cmd

    # ── API ────────────────────────────────────────────────────────────
    def reset(self, *, command: np.ndarray | None = None) -> np.ndarray:
        m, d = self.model, self.data
        self.model_params = self.dr_override or dr_mod.sample_model_params(self.rng, self.cfg.dr)
        self.ep = self.episode_override or dr_mod.sample_episode_params(self.rng, self.cfg.dr)
        dr_mod.apply_model_params_mujoco(m, None, self.model_params, self.floor_id, self.torso_id,
                                         self._base_mass, self._base_ipos, self._base_damping, self._base_frictionloss)
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        # initial pose noise
        self.target_deg = self.stand_deg + self.ep.init_joint_noise_deg
        self.target_deg = np.clip(np.round(self.target_deg), self.lo, self.hi)
        ctrl = jm.policy_to_ctrl(jm.agent_deg_to_mjcf_rad(self.target_deg))
        d.qpos[self.qpos_idx] = jm.agent_deg_to_mjcf_rad(self.target_deg)
        roll, pitch = self.ep.init_tilt_rad
        q = _rpy_to_quat(roll, pitch, 0.0)
        d.qpos[3:7] = q
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        self.hist = np.tile(self.target_deg, (max(self.history_n, spec.MAX_LATENCY + 1), 1))
        self.last_action = np.zeros(8)
        self.action = np.zeros(8)
        self.feet_air_time = np.zeros(4)
        self.last_contact = np.zeros(4, dtype=bool)
        self.body_contact_steps = 0
        self.t = 0
        self.fixed_command = command
        self.cmd = np.array(command, dtype=np.float64) if command is not None else self._sample_command()
        self.steps_until_cmd = int(round(self.cfg.commands.resample_s / self.cfg.control_dt))
        self.next_push_step = int(round(self.ep.push_interval_s / self.cfg.control_dt)) if self.ep.push_enabled else -1
        self.reset_count += 1
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, StepInfo]:
        m, d = self.model, self.data
        self.last_action = self.action
        self.action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.target_deg = spec.process_action(np, self.action, self.target_deg, self.cfg.action_scale_deg,
                                              self.lo, self.hi, self.cfg.action_mode)
        # history newest-first; latency picks an older commanded target
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = self.target_deg
        applied = self.hist[self.ep.latency_steps]
        d.ctrl[:] = jm.policy_to_ctrl(jm.agent_deg_to_mjcf_rad(applied))
        if self.next_push_step >= 0 and self.t == self.next_push_step:
            ang = self.rng.uniform(0, 2 * np.pi)
            d.qvel[0] += self.ep.push_vel * np.cos(ang)
            d.qvel[1] += self.ep.push_vel * np.sin(ang)
            self.next_push_step = self.t + int(round(self.ep.push_interval_s / self.cfg.control_dt))
        torques = np.zeros(8)
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
            torques += np.abs(d.actuator_force)
        torques /= self.n_substeps
        self.t += 1

        contact = np.array([self.sensor(n)[0] > 0 for n in self.foot_found])
        contact_filt = contact | self.last_contact
        first_contact = (self.feet_air_time > 0) & contact_filt
        self.feet_air_time += self.cfg.control_dt
        body_contact = any(self.sensor(n)[0] > 0 for n in self.body_found)
        self.body_contact_steps = self.body_contact_steps + 1 if body_contact else 0

        q = self._quantities(torques, contact, first_contact)
        terms = spec.reward_terms(np, q, self.cfg.reward.tracking_sigma, self.cfg.reward.ang_tracking_sigma,
                                  self.cfg.reward.base_height_target)
        weights = self.cfg.reward.weights.model_dump()
        reward = float(spec.weighted_reward(np, terms, weights, self.cfg.control_dt))
        done = bool(spec.termination(np, q["up_z"], q["torso_z"], self.body_contact_steps))

        self.feet_air_time *= ~contact
        self.last_contact = contact
        if self.fixed_command is None:
            self.steps_until_cmd -= 1
            if self.steps_until_cmd <= 0:
                self.cmd = self._sample_command()
                self.steps_until_cmd = int(round(self.cfg.commands.resample_s / self.cfg.control_dt))
        info = StepInfo(terms={k: float(v) for k, v in terms.items()}, contact=contact,
                        target_deg=self.target_deg.copy(), applied_deg=applied.copy(), fell=done)
        return self._obs(), reward, done, info

    # ── observations / quantities ──────────────────────────────────────
    def _gravity(self) -> np.ndarray:
        return spec.gravity_from_xmat(np, self.data.site_xmat[self.imu_site].reshape(3, 3))

    def _gyro(self) -> np.ndarray:
        return self.sensor("imu_gyro") + self.ep.gyro_bias

    def _obs(self) -> np.ndarray:
        gravity = self._gravity()
        gyro = self._gyro()
        if self.cfg.dr.enabled:
            gravity = gravity + self.rng.uniform(-1, 1, 3) * self.cfg.dr.gravity_noise
            gyro = gyro + self.rng.uniform(-1, 1, 3) * self.cfg.dr.gyro_noise
        phase = None
        if self.cfg.obs.phase_clock:
            ph = 2 * np.pi * (self.t * self.cfg.control_dt) / 0.5
            phase = np.array([np.sin(ph), np.cos(ph)])
        return spec.build_obs(np, gravity, gyro, self.cmd, self.hist[: self.history_n], self.action, phase)

    def _quantities(self, torques, contact, first_contact) -> dict[str, Any]:
        d = self.data
        feet_vel = np.stack([self.sensor(n) for n in self.foot_vel])
        return {
            "cmd": self.cmd,
            "local_linvel": self.sensor("torso_vel"),
            "gyro": self.sensor("imu_gyro"),
            "global_linvel": self.sensor("torso_global_linvel"),
            "global_angvel": self.sensor("torso_global_angvel"),
            "gravity": self._gravity(),
            "up_z": float(self.sensor("torso_upvector")[2]),
            "torso_z": float(d.qpos[2]),
            "action": self.action,
            "last_action": self.last_action,
            "torques": torques,
            "joint_vel": d.qvel[self.dof_idx],
            "target_norm": spec.target_saturation(np, self.target_deg, self.lo, self.hi),
            "target_deg": self.target_deg,
            "feet_air_time": self.feet_air_time,
            "first_contact": first_contact.astype(np.float64),
            "contact": contact.astype(np.float64),
            "feet_vel_xy": feet_vel[:, :2],
        }

    # ── recording helpers ──────────────────────────────────────────────
    def base_state(self) -> dict[str, Any]:
        d = self.data
        q = d.qpos[3:7]
        return {
            "pos": d.qpos[0:3].tolist(),
            "quat_wxyz": q.tolist(),
            "rpy_deg": [float(np.degrees(x)) for x in _quat_to_rpy(q)],
            "joint_deg": jm.mjcf_rad_to_agent_deg(d.qpos[self.qpos_idx]).tolist(),
        }

    def set_ctrl_deg(self, target_deg: np.ndarray) -> None:
        """Bypass the policy: drive commanded targets directly (gait replay)."""
        self.target_deg = np.clip(np.round(np.asarray(target_deg, dtype=np.float64)), self.lo, self.hi)


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)
