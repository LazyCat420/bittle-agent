"""Seeded episode evaluation in the CPU MuJoCo env.

``run_episode`` drives an env with a *controller* (a trained policy, a
replayed OpenCat gait, or "stand") and returns per-episode statistics; the
``aggregate`` helper turns a list of them into the metrics the gates read.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from ..assets import joint_map as jm
from ..assets import opencat_gaits
from ..config import TrainConfig
from ..env import dr as dr_mod
from ..env import spec
from ..env.cpu_env import BittleCpuEnv, _quat_to_rpy
from .rollout import RolloutRecorder

SAT_MARGIN_DEG = 2.0


@dataclass
class EpisodeStats:
    seed: int
    seconds: float
    steps: int
    cmd: list[float]
    distance_x: float
    lateral_y: float
    yaw_deg: float
    fell: bool
    fall_time: float | None
    vel_rmse: float
    saturation_pct: float
    smoothness_deg: float
    tilt_deg: float
    rms_vz: float
    energy_w: float
    mean_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── controllers ────────────────────────────────────────────────────────────

class PolicyController:
    def __init__(self, policy: Callable[[np.ndarray], np.ndarray]):
        self.policy = policy

    def reset(self, env: BittleCpuEnv) -> None:
        pass

    def act(self, obs: np.ndarray, env: BittleCpuEnv, t: int) -> tuple[str, np.ndarray]:
        return "action", np.asarray(self.policy(obs), dtype=np.float64)


class StandController:
    def reset(self, env: BittleCpuEnv) -> None:
        pass

    def act(self, obs, env, t):
        return "target", spec.STAND_DEG.copy()


class GaitController:
    """Replay an OpenCat gait table (or keyframe moveset) as commanded targets."""

    def __init__(self, targets_deg: np.ndarray, *, rows_per_step: float = 1.0, lead_in_steps: int = 25):
        self.targets = np.asarray(targets_deg, dtype=np.float64)  # (n, 8) policy order, agent degrees
        self.rows_per_step = rows_per_step
        self.lead_in = lead_in_steps

    @classmethod
    def from_opencat(cls, name: str, **kw) -> "GaitController":
        ctrl_rows = np.array(opencat_gaits.gait_ctrl_frames(name))  # (n, 8) ctrl order, rad
        policy_rows = jm.ctrl_to_policy(ctrl_rows)
        deg = np.array([jm.mjcf_rad_to_agent_deg(r) for r in policy_rows])
        return cls(deg, **kw)

    @classmethod
    def from_keyframes(cls, frames: list[dict[str, Any]], control_hz: int = 50, **kw) -> "GaitController":
        """bittle-agent moveset frames ({angles, delay_ms}) -> per-step targets (linear slew)."""
        rows = []
        prev = spec.STAND_DEG.copy()
        for f in frames:
            angles = f.get("angles", {})
            tgt = np.array([float(angles.get(i, angles.get(str(i), prev[k]))) for k, i in enumerate(jm.POLICY_JOINTS)])
            n = max(1, int(round(float(f.get("delay_ms", 200)) / 1000.0 * control_hz)))
            for s in range(1, n + 1):
                rows.append(prev + (tgt - prev) * (s / n))
            prev = tgt
        return cls(np.array(rows), **kw)

    def reset(self, env: BittleCpuEnv) -> None:
        pass

    def act(self, obs, env, t):
        if t < self.lead_in:
            return "target", spec.STAND_DEG.copy()
        idx = int((t - self.lead_in) * self.rows_per_step) % len(self.targets)
        return "target", self.targets[idx]


# ── episode loop ───────────────────────────────────────────────────────────

def run_episode(env: BittleCpuEnv, controller, *, seed: int, seconds: float, command: np.ndarray,
                recorder: RolloutRecorder | None = None) -> EpisodeStats:
    env.rng = np.random.default_rng(seed)
    obs = env.reset(command=np.asarray(command, dtype=np.float64))
    controller.reset(env)
    steps = int(round(seconds * env.cfg.control_hz))
    dt = env.cfg.control_dt
    x0, y0 = float(env.data.qpos[0]), float(env.data.qpos[1])
    lo, hi = env.lo, env.hi
    vel_err2 = 0.0
    sat = 0
    smooth = 0.0
    tilt = 0.0
    vz2 = 0.0
    energy = 0.0
    rewards = 0.0
    prev_target = env.target_deg.copy()
    fell = False
    fall_time = None
    n = 0
    for t in range(steps):
        kind, val = controller.act(obs, env, t)
        if kind == "target":
            obs, r, done, info = env.step(np.zeros(8), target_deg=val)
        else:
            obs, r, done, info = env.step(val)
        n += 1
        rewards += r
        vx = float(env.sensor("torso_vel")[0])
        vel_err2 += (vx - float(command[0])) ** 2
        sat += int(np.sum((info.target_deg <= lo + SAT_MARGIN_DEG) | (info.target_deg >= hi - SAT_MARGIN_DEG)))
        smooth += float(np.mean(np.abs(info.target_deg - prev_target)))
        prev_target = info.target_deg.copy()
        up_z = float(np.clip(env.sensor("torso_upvector")[2], -1, 1))
        tilt += math.degrees(math.acos(up_z))
        vz2 += float(env.sensor("torso_global_linvel")[2]) ** 2
        energy += float(np.sum(np.abs(env.data.actuator_force * env.data.qvel[env.dof_idx])))
        if recorder is not None:
            recorder.record((t + 1) * dt, env, info.applied_deg, info.contact, env.cmd)
        if done:
            fell = True
            fall_time = (t + 1) * dt
            if recorder is not None:
                recorder.event(fall_time, "fall")
            break
    q = env.data.qpos[3:7]
    yaw = math.degrees(_quat_to_rpy(q)[2])
    return EpisodeStats(
        seed=seed, seconds=n * dt, steps=n, cmd=[float(c) for c in command],
        distance_x=float(env.data.qpos[0] - x0), lateral_y=float(env.data.qpos[1] - y0), yaw_deg=yaw,
        fell=fell, fall_time=fall_time,
        vel_rmse=math.sqrt(vel_err2 / max(n, 1)), saturation_pct=100.0 * sat / max(n * 8, 1),
        smoothness_deg=smooth / max(n, 1), tilt_deg=tilt / max(n, 1), rms_vz=math.sqrt(vz2 / max(n, 1)),
        energy_w=energy / max(n, 1), mean_reward=rewards / max(n, 1),
    )


def aggregate(stats: list[EpisodeStats], episode_seconds: float) -> dict[str, Any]:
    if not stats:
        return {}
    d = np.array([s.distance_x for s in stats])
    falls = [s for s in stats if s.fell]
    return {
        "n_episodes": len(stats),
        "episode_seconds": episode_seconds,
        "fall_rate": len(falls) / len(stats),
        "first_fall_time_mean": float(np.mean([s.fall_time for s in falls])) if falls else None,
        "forward_distance_p50": float(np.median(d)),
        "forward_distance_mean": float(np.mean(d)),
        "forward_distance_min": float(np.min(d)),
        "vel_tracking_rmse": float(np.mean([s.vel_rmse for s in stats])),
        "heading_yaw_deg": float(np.mean([abs(s.yaw_deg) for s in stats])),
        "lateral_drift_m": float(np.mean([abs(s.lateral_y) for s in stats])),
        "joint_saturation_pct": float(np.mean([s.saturation_pct for s in stats])),
        "action_smoothness_deg": float(np.mean([s.smoothness_deg for s in stats])),
        "mean_tilt_deg": float(np.mean([s.tilt_deg for s in stats])),
        "rms_vz": float(np.mean([s.rms_vz for s in stats])),
        "energy_proxy_w": float(np.mean([s.energy_w for s in stats])),
        "mean_reward": float(np.mean([s.mean_reward for s in stats])),
    }


# ── protocols ──────────────────────────────────────────────────────────────

def nominal_env(cfg: TrainConfig, *, variant: str = "cpu", envelope_tier: str = "agent",
                model_params: dr_mod.ModelParams | None = None,
                episode_params: dr_mod.EpisodeParams | None = None) -> BittleCpuEnv:
    """Env with DR fixed to nominal (or the given overrides) — the benchmark protocol."""
    return BittleCpuEnv(cfg, variant=variant, seed=0, envelope_tier=envelope_tier,
                        dr_override=model_params or dr_mod.nominal_model_params(),
                        episode_override=episode_params or dr_mod.nominal_episode_params())


DR_PRESETS: dict[str, dict[str, Any]] = {
    "friction_low": {"friction": 0.5},
    "friction_high": {"friction": 1.1},
    "payload_30g": {"payload_kg": 0.03},
    "kp_low": {"kp": 28.0},
    "latency_3": {"latency_steps": 3},
}


def preset_params(name: str) -> tuple[dr_mod.ModelParams, dr_mod.EpisodeParams]:
    mp = dr_mod.nominal_model_params()
    ep = dr_mod.nominal_episode_params()
    for k, v in DR_PRESETS[name].items():
        if hasattr(mp, k):
            setattr(mp, k, v)
        else:
            setattr(ep, k, v)
    return mp, ep


def evaluate_protocol(cfg: TrainConfig, controller, *, n_episodes: int, seed_start: int, seconds: float,
                      command_vx: float, variant: str = "cpu", envelope_tier: str = "agent",
                      model_params=None, episode_params=None, record_seeds: set[int] | None = None,
                      source: dict[str, Any] | None = None) -> tuple[list[EpisodeStats], dict[int, dict[str, Any]]]:
    env = nominal_env(cfg, variant=variant, envelope_tier=envelope_tier, model_params=model_params,
                      episode_params=episode_params)
    cmd = np.array([command_vx, 0.0, 0.0])
    stats, rollouts = [], {}
    for i in range(n_episodes):
        seed = seed_start + i
        rec = RolloutRecorder(cfg.control_hz, dict(source or {}, seed=seed, sim=f"mujoco-cpu/{variant}")) if record_seeds and seed in record_seeds else None
        st = run_episode(env, controller, seed=seed, seconds=seconds, command=cmd, recorder=rec)
        stats.append(st)
        if rec is not None:
            rollouts[seed] = rec.to_dict({"distance_m": round(st.distance_x, 4), "fell": st.fell,
                                          "fall_time": st.fall_time, "seconds": st.seconds})
    return stats, rollouts
