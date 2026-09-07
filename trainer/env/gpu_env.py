"""GPU environment (MuJoCo Warp by default, MJX/JAX behind ``impl="jax"``).

A ``mujoco_playground`` ``MjxEnv`` implementing the shared ``spec.py``
with ``jax.numpy``. Model-level domain randomisation is the Playground
pattern (a vmapped ``randomization_fn`` over the model); episode-level DR
(latency, gyro bias, init noise, pushes) lives in ``state.info``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from ..assets import joint_map as jm
from ..config import TrainConfig
from . import spec

GENERATED = Path(__file__).resolve().parent.parent / "assets" / "generated"


def playground_config(cfg: TrainConfig, *, impl: str | None = None, num_envs: int = 1,
                      sim_dt: float = 0.002) -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=cfg.control_dt,
        sim_dt=sim_dt,
        episode_length=cfg.episode_steps,
        action_repeat=cfg.ppo.action_repeat,
        impl=impl or cfg.sim_impl,
        naconmax=int(16 * max(num_envs, 1)),
        njmax=96,
    )


class BittleGpuEnv(mjx_env.MjxEnv):
    def __init__(self, config: TrainConfig, *, impl: str | None = None, num_envs: int = 1,
                 variant: str = "gpu", sim_dt: float = 0.002):
        self.cfg = config.resolved()
        pg = playground_config(self.cfg, impl=impl, num_envs=num_envs, sim_dt=sim_dt)
        super().__init__(pg)
        self._xml_path = str(GENERATED / f"bittle_{variant}.xml")
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        m = self._mj_model

        self._qpos_idx = jp.array(jm.qpos_indices(m))
        self._dof_idx = jp.array(jm.dof_indices(m))
        self._torso_id = m.body("torso").id
        self._imu_site = m.site("imu_site").id
        lo, hi = spec.load_envelope("agent")
        self._lo, self._hi = jp.array(lo), jp.array(hi)
        self._stand_deg = jp.array(spec.STAND_DEG)
        self._sign = jp.array(jm.SIGN_POLICY)
        self._offset = jp.array(jm.OFFSET_POLICY_DEG)
        self._p2c = jp.array(jm.POLICY_TO_CTRL)
        self._init_q = jp.array(m.key_qpos[0])
        self._history_n = self.cfg.obs.history_n
        self._hist_len = max(self._history_n, spec.MAX_LATENCY + 1)
        self._weights = {k: float(v) for k, v in self.cfg.reward.weights.model_dump().items()}

        def sadr(name: str) -> tuple[int, int]:
            sid = m.sensor(name).id
            return int(m.sensor_adr[sid]), int(m.sensor_dim[sid])

        self._s = {n: sadr(n) for n in ["imu_gyro", "torso_vel", "torso_global_linvel", "torso_global_angvel",
                                        "torso_upvector"]}
        self._foot_found = [sadr(f"{leg}_foot_floor_found")[0] for leg in jm.LEGS]
        self._body_found = [int(m.sensor_adr[i]) for i in range(m.nsensor)
                            if m.sensor(i).name.endswith("_floor_found") and "foot" not in m.sensor(i).name]
        self._foot_vel = jp.array([list(range(sadr(f"{leg}_foot_global_linvel")[0], sadr(f"{leg}_foot_global_linvel")[0] + 3))
                                   for leg in jm.LEGS])
        self._cmd_lo = jp.array([self.cfg.commands.vx[0], self.cfg.commands.vy[0], self.cfg.commands.wz[0]])
        self._cmd_hi = jp.array([self.cfg.commands.vx[1], self.cfg.commands.vy[1], self.cfg.commands.wz[1]])
        self._resample_steps = int(round(self.cfg.commands.resample_s / self.dt))
        self._dr = self.cfg.dr

    # ── conversions ────────────────────────────────────────────────────
    def _deg_to_ctrl(self, deg_policy: jax.Array) -> jax.Array:
        rad = (deg_policy + self._offset) * self._sign * (math.pi / 180.0)
        return jp.zeros(8).at[self._p2c].set(rad)

    def _sensor(self, data: mjx.Data, name: str) -> jax.Array:
        adr, dim = self._s[name]
        return data.sensordata[adr:adr + dim]

    def _gravity(self, data: mjx.Data) -> jax.Array:
        return spec.gravity_from_xmat(jp, data.site_xmat[self._imu_site])

    # ── episode sampling ───────────────────────────────────────────────
    def _sample_episode(self, rng: jax.Array) -> dict[str, Any]:
        dr = self._dr
        k = jax.random.split(rng, 8)
        en = 1.0 if dr.enabled else 0.0
        latency = jax.random.randint(k[0], (), dr.latency_steps[0], dr.latency_steps[1] + 1)
        latency = jp.where(dr.enabled, latency, 1)
        return {
            "latency": latency,
            "gyro_bias": en * jax.random.uniform(k[1], (3,), minval=-dr.gyro_bias, maxval=dr.gyro_bias),
            "init_joint_noise": en * jax.random.uniform(k[2], (8,), minval=-dr.init_joint_noise_deg, maxval=dr.init_joint_noise_deg),
            "init_tilt": en * jax.random.uniform(k[3], (2,), minval=-dr.init_tilt_rad, maxval=dr.init_tilt_rad),
            "push_vel": jax.random.uniform(k[4], (), minval=dr.push_vel[0], maxval=dr.push_vel[1]),
            "push_steps": jp.round(jax.random.uniform(k[5], (), minval=dr.push_interval_s[0], maxval=dr.push_interval_s[1]) / self.dt).astype(jp.int32),
        }

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        k1, k2 = jax.random.split(rng)
        cmd = jax.random.uniform(k1, (3,), minval=self._cmd_lo, maxval=self._cmd_hi)
        zero = jax.random.uniform(k2) < spec.ZERO_CMD_PROB
        return jp.where(zero, jp.zeros(3), cmd)

    # ── MjxEnv API ─────────────────────────────────────────────────────
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_ep, k_cmd, k_push = jax.random.split(rng, 4)
        ep = self._sample_episode(k_ep)
        target = jp.clip(jp.round(self._stand_deg + ep["init_joint_noise"]), self._lo, self._hi)
        rad = (target + self._offset) * self._sign * (math.pi / 180.0)
        qpos = self._init_q.at[self._qpos_idx].set(rad)
        roll, pitch = ep["init_tilt"][0], ep["init_tilt"][1]
        qpos = qpos.at[3:7].set(_rpy_to_quat(roll, pitch, 0.0))
        ctrl = self._deg_to_ctrl(target)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=jp.zeros(self.mjx_model.nv), ctrl=ctrl,
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "command": self._sample_command(k_cmd),
            "steps_until_cmd": jp.int32(self._resample_steps),
            "target_deg": target,
            "hist": jp.tile(target, (self._hist_len, 1)),
            "last_act": jp.zeros(8),
            "act": jp.zeros(8),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "body_contact_steps": jp.int32(0),
            "step": jp.int32(0),
            "prev_x": qpos[0],
            "next_push": jp.where(self._dr.push_enabled, ep["push_steps"], jp.int32(-1)),
            "latency": ep["latency"],
            "gyro_bias": ep["gyro_bias"],
            "push_vel": ep["push_vel"],
            "push_steps": ep["push_steps"],
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._weights}
        metrics["distance_x"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        act = jp.clip(action, -1.0, 1.0)
        target = spec.process_action(jp, act, info["target_deg"], self.cfg.action_scale_deg, self._lo, self._hi,
                                     self.cfg.action_mode)
        hist = jp.roll(info["hist"], 1, axis=0).at[0].set(target)
        applied = hist[info["latency"]]
        ctrl = self._deg_to_ctrl(applied)

        data = state.data
        # push: velocity kick on the base at scheduled steps (stage >= 2)
        do_push = (info["next_push"] >= 0) & (info["step"] == info["next_push"])
        info["rng"], k_push = jax.random.split(info["rng"])
        ang = jax.random.uniform(k_push, (), minval=0.0, maxval=2 * jp.pi)
        kick = jp.where(do_push, info["push_vel"], 0.0) * jp.array([jp.cos(ang), jp.sin(ang)])
        data = data.replace(qvel=data.qvel.at[0:2].add(kick))
        next_push = jp.where(do_push, info["step"] + info["push_steps"], info["next_push"])

        data = mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)

        contact = jp.array([data.sensordata[a] > 0 for a in self._foot_found])
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) & contact_filt
        feet_air_time = info["feet_air_time"] + self.dt
        body_contact = jp.any(jp.array([data.sensordata[a] > 0 for a in self._body_found])) if self._body_found else jp.bool_(False)
        body_steps = jp.where(body_contact, info["body_contact_steps"] + 1, 0)

        torques = jp.abs(data.actuator_force)
        q = {
            "cmd": info["command"],
            "local_linvel": self._sensor(data, "torso_vel"),
            "gyro": self._sensor(data, "imu_gyro"),
            "global_linvel": self._sensor(data, "torso_global_linvel"),
            "global_angvel": self._sensor(data, "torso_global_angvel"),
            "gravity": self._gravity(data),
            "up_z": self._sensor(data, "torso_upvector")[2],
            "torso_z": data.qpos[2],
            "action": act,
            "last_action": info["act"],
            "torques": torques,
            "joint_vel": data.qvel[self._dof_idx],
            "target_norm": spec.target_saturation(jp, target, self._lo, self._hi),
            "target_deg": target,
            "feet_air_time": feet_air_time,
            "first_contact": first_contact.astype(jp.float32),
            "contact": contact.astype(jp.float32),
            "feet_vel_xy": data.sensordata[self._foot_vel][:, :2],
        }
        terms = spec.reward_terms(jp, q, self.cfg.reward.tracking_sigma, self.cfg.reward.ang_tracking_sigma,
                                  self.cfg.reward.base_height_target)
        reward = spec.weighted_reward(jp, terms, self._weights, self.dt)
        done = spec.termination(jp, q["up_z"], q["torso_z"], body_steps)

        # bookkeeping
        info["rng"], k_cmd = jax.random.split(info["rng"])
        steps_until = info["steps_until_cmd"] - 1
        info["command"] = jp.where(steps_until <= 0, self._sample_command(k_cmd), info["command"])
        info["steps_until_cmd"] = jp.where(steps_until <= 0, self._resample_steps, steps_until)
        info["target_deg"] = target
        info["hist"] = hist
        info["last_act"] = info["act"]
        info["act"] = act
        info["feet_air_time"] = feet_air_time * (~contact)
        info["last_contact"] = contact
        info["body_contact_steps"] = body_steps
        info["step"] = info["step"] + 1
        info["next_push"] = next_push
        for k, v in terms.items():
            state.metrics[f"reward/{k}"] = v * self._weights[k]
        # per-step displacement, so brax's episode SUM of this metric is the distance walked
        state.metrics["distance_x"] = data.qpos[0] - info["prev_x"]
        info["prev_x"] = data.qpos[0]

        obs = self._get_obs(data, info)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
        en = 1.0 if self._dr.enabled else 0.0
        gravity = self._gravity(data) + en * jax.random.uniform(k1, (3,), minval=-1, maxval=1) * self._dr.gravity_noise
        gyro = self._sensor(data, "imu_gyro") + info["gyro_bias"] + en * jax.random.uniform(k2, (3,), minval=-1, maxval=1) * self._dr.gyro_noise
        phase = None
        if self.cfg.obs.phase_clock:
            ph = 2 * jp.pi * (info["step"] * self.dt) / 0.5
            phase = jp.array([jp.sin(ph), jp.cos(ph)])
        state = spec.build_obs(jp, gravity, gyro, info["command"], info["hist"][: self._history_n], info["act"], phase)
        privileged = jp.concatenate([
            state,
            self._sensor(data, "torso_vel"),
            self._sensor(data, "torso_global_angvel"),
            data.qpos[self._qpos_idx],
            data.qvel[self._dof_idx],
            info["last_contact"].astype(jp.float32),
            info["feet_air_time"],
            jp.array([data.qpos[2]]),
        ]).astype(jp.float32)
        return {"state": state, "privileged_state": privileged}

    # ── accessors ──────────────────────────────────────────────────────
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 8

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def obs_layout(self) -> list[tuple[str, int]]:
        return spec.obs_layout(self._history_n, self.cfg.obs.phase_clock)


def _rpy_to_quat(roll, pitch, yaw):
    cr, sr = jp.cos(roll / 2), jp.sin(roll / 2)
    cp, sp = jp.cos(pitch / 2), jp.sin(pitch / 2)
    cy, sy = jp.cos(yaw / 2), jp.sin(yaw / 2)
    return jp.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def make_domain_randomizer(cfg: TrainConfig, mj_model: mujoco.MjModel):
    """Playground-style ``randomization_fn``: per-env model params, fixed for the run."""
    dr = cfg.resolved().dr
    floor_id = mj_model.geom("floor").id
    torso_id = mj_model.body("torso").id

    def domain_randomize(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand(rng):
            k = jax.random.split(rng, 8)
            friction = model.geom_friction.at[floor_id, 0].set(jax.random.uniform(k[0], minval=dr.friction[0], maxval=dr.friction[1]))
            mass = model.body_mass * jax.random.uniform(k[1], minval=dr.mass_scale[0], maxval=dr.mass_scale[1])
            mass = mass.at[torso_id].add(jax.random.uniform(k[2], minval=dr.payload_g[0], maxval=dr.payload_g[1]) / 1000.0)
            shift = jp.array([jax.random.uniform(k[3], minval=dr.com_shift_mm[0], maxval=dr.com_shift_mm[1]) / 1000.0, 0.0, 0.0])
            ipos = model.body_ipos.at[torso_id].set(model.body_ipos[torso_id] + shift)
            kp = jax.random.uniform(k[4], minval=dr.kp[0], maxval=dr.kp[1])
            gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
            fr = jax.random.uniform(k[5], minval=dr.forcerange[0], maxval=dr.forcerange[1])
            forcerange = jp.stack([-fr * jp.ones(model.nu), fr * jp.ones(model.nu)], axis=1)
            damping = model.dof_damping.at[6:].set(jax.random.uniform(k[6], minval=dr.damping[0], maxval=dr.damping[1]))
            frictionloss = model.dof_frictionloss.at[6:].set(jax.random.uniform(k[7], minval=dr.frictionloss[0], maxval=dr.frictionloss[1]))
            return friction, mass, ipos, gainprm, biasprm, forcerange, damping, frictionloss

        friction, mass, ipos, gainprm, biasprm, forcerange, damping, frictionloss = rand(rng)
        fields = {
            "geom_friction": friction, "body_mass": mass, "body_ipos": ipos,
            "actuator_gainprm": gainprm, "actuator_biasprm": biasprm, "actuator_forcerange": forcerange,
            "dof_damping": damping, "dof_frictionloss": frictionloss,
        }
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({k: 0 for k in fields})
        model = model.tree_replace(fields)
        return model, in_axes

    return domain_randomize if dr.enabled else None
