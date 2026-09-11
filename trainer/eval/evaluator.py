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
from ..env import terrain as tr
from ..env.cpu_env import BittleCpuEnv, _quat_to_rpy
from .rollout import RolloutRecorder

SAT_MARGIN_DEG = 2.0
#: "stuck": commanded to move but |vx| below STUCK_VX for at least STUCK_MIN_S (10 mm/s over 0.5 s)
STUCK_VX = 0.02
STUCK_MIN_S = 0.5


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
    # ── terrain / servo-safety (2026-09-07) ──
    ang_vel_rmse: float = 0.0          # gyro_z vs commanded wz (rad/s)
    centre_drift_m: float = 0.0        # |xy displacement| (a spin should stay put)
    climb_height: float = 0.0          # displacement along the uphill direction x sin(slope)
    stumble_rate: float = 0.0          # fraction of steps with a shank/thigh on the ground
    foot_clearance_p50_mm: float = 0.0  # median swing-foot height above the local terrain
    body_clearance_min_mm: float = 0.0  # min torso height above the local terrain
    climb_max_m: float = 0.0           # highest support height reached (stairs: risers x steps climbed)
    descent_m: float = 0.0             # climb_max - final support height (stairs: how far it came back down)
    peak_joint_speed_rad_s: float = 0.0  # p99.5 of substep |qvel| over (step, joint)
    max_joint_speed_rad_s: float = 0.0
    stall_fraction: float = 0.0        # (control step, joint) pairs stalled for the whole step / all pairs
    stall_concurrent_max: int = 0      # most joints stalled in the same control step
    travel_deg_max: float = 0.0        # max over joints of sum |delta target|
    scene: str = ""                    # the suite scene this episode ran in ("" = single-protocol suite)
    # ── stuck diagnostics (2026-09-10): WHERE and for how long the robot stopped while commanded ──
    stuck_seconds: float = 0.0         # time inside stalls of >= STUCK_MIN_S with |vx| < STUCK_VX (commanded to move)
    stuck_x: float | None = None       # x of the first such stall (None = never stuck)
    stuck_limb_steps: int = 0          # steps inside those stalls with a shank/thigh on the ground (edge catch)

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

    #: OpenCat plays gait rows at roughly 100 Hz (a 48-row trot cycle in ~0.5 s); at 50 Hz control
    #: that is two rows per step. At one row per step the open-loop trot slips backwards.
    DEFAULT_ROWS_PER_STEP = 2.0

    def __init__(self, targets_deg: np.ndarray, *, rows_per_step: float | None = None, lead_in_steps: int = 25):
        self.targets = np.asarray(targets_deg, dtype=np.float64)  # (n, 8) policy order, agent degrees
        self.rows_per_step = self.DEFAULT_ROWS_PER_STEP if rows_per_step is None else rows_per_step
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
    uphill = tr.uphill_xy(np, env.model.opt.gravity)
    ang_err2 = 0.0
    stumbles = 0
    swing_peaks: list[float] = []      # peak clearance of each completed swing phase
    swing_cur = np.full(4, -math.inf)  # running max while a foot is off the ground
    in_swing = np.zeros(4, dtype=bool)
    body_clear_min = math.inf
    support_max, support_last = 0.0, 0.0
    peaks: list[np.ndarray] = []
    stall_joint_steps = 0
    stall_conc = 0
    travel = np.zeros(8)
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
    commanded = abs(float(command[0])) >= spec.ZERO_CMD_EPS
    stuck_min_steps = max(1, int(round(STUCK_MIN_S / dt)))
    slow_run = 0            # consecutive slow steps
    slow_limb = 0           # of which with a limb on the ground
    slow_x0 = 0.0           # x where the current slow stretch began
    stuck_steps = 0
    stuck_limb = 0
    stuck_x = None
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
        ang_err2 += (float(env.sensor("imu_gyro")[2]) - float(command[2])) ** 2
        sat += int(np.sum((info.target_deg <= lo + SAT_MARGIN_DEG) | (info.target_deg >= hi - SAT_MARGIN_DEG)))
        smooth += float(np.mean(np.abs(info.target_deg - prev_target)))
        travel += np.abs(info.target_deg - prev_target)
        prev_target = info.target_deg.copy()
        limb_on = int(info.limb_contact.sum() > 0)
        stumbles += limb_on
        if commanded and abs(vx) < STUCK_VX:
            if slow_run == 0:
                slow_x0 = float(env.data.qpos[0])
            slow_run += 1
            slow_limb += limb_on
            if slow_run == stuck_min_steps:     # the stretch just became a stall: count it from its start
                stuck_steps += slow_run
                stuck_limb += slow_limb
                if stuck_x is None:
                    stuck_x = slow_x0
            elif slow_run > stuck_min_steps:
                stuck_steps += 1
                stuck_limb += limb_on
        else:
            slow_run, slow_limb = 0, 0
        for i in range(4):
            if not info.contact[i]:
                in_swing[i] = True
                swing_cur[i] = max(swing_cur[i], float(info.foot_clearance[i]))
            elif in_swing[i]:
                swing_peaks.append(swing_cur[i])
                in_swing[i], swing_cur[i] = False, -math.inf
        body_clear_min = min(body_clear_min, float(env.data.qpos[2]) - info.terrain_h)
        support_max = max(support_max, info.terrain_h)
        support_last = info.terrain_h
        peaks.append(info.peak_qvel)
        n_st = int(info.stalled.sum())
        stall_joint_steps += n_st
        stall_conc = max(stall_conc, n_st)
        up_z = float(np.clip(env.sensor("torso_upvector")[2], -1, 1))
        tilt += math.degrees(math.acos(up_z))
        vz2 += float(env.sensor("torso_global_linvel")[2]) ** 2
        energy += float(np.sum(np.abs(env.data.actuator_force * env.data.qvel[env.dof_idx])))
        if recorder is not None:
            recorder.record((t + 1) * dt, env, info.applied_deg, info.contact, env.cmd, limb_contact=info.limb_contact)
        if done:
            fell = True
            fall_time = (t + 1) * dt
            if recorder is not None:
                recorder.event(fall_time, "fall")
            break
    q = env.data.qpos[3:7]
    yaw = math.degrees(_quat_to_rpy(q)[2])
    disp = np.array([env.data.qpos[0] - x0, env.data.qpos[1] - y0])
    peak_arr = np.concatenate(peaks) if peaks else np.zeros(1)
    n_samples = max(n * 8, 1)
    return EpisodeStats(
        ang_vel_rmse=math.sqrt(ang_err2 / max(n, 1)), centre_drift_m=float(np.linalg.norm(disp)),
        climb_height=float(np.dot(disp, uphill)), stumble_rate=stumbles / max(n, 1),
        foot_clearance_p50_mm=1000.0 * float(np.median(swing_peaks)) if swing_peaks else 0.0,
        body_clearance_min_mm=1000.0 * (body_clear_min if math.isfinite(body_clear_min) else 0.0),
        climb_max_m=float(support_max), descent_m=float(max(support_max - support_last, 0.0)),
        peak_joint_speed_rad_s=float(np.percentile(peak_arr, 99.5)), max_joint_speed_rad_s=float(peak_arr.max()),
        stall_fraction=stall_joint_steps / n_samples, stall_concurrent_max=int(stall_conc),
        travel_deg_max=float(travel.max()),
        stuck_seconds=stuck_steps * dt, stuck_x=stuck_x, stuck_limb_steps=int(stuck_limb),
        seed=seed, seconds=n * dt, steps=n, cmd=[float(c) for c in command],
        distance_x=float(env.data.qpos[0] - x0), lateral_y=float(env.data.qpos[1] - y0), yaw_deg=yaw,
        fell=fell, fall_time=fall_time,
        vel_rmse=math.sqrt(vel_err2 / max(n, 1)), saturation_pct=100.0 * sat / max(n * 8, 1),
        smoothness_deg=smooth / max(n, 1), tilt_deg=tilt / max(n, 1), rms_vz=math.sqrt(vz2 / max(n, 1)),
        energy_w=energy / max(n, 1), mean_reward=rewards / max(n, 1),
    )


def _stuck_limb_share(stats: list[EpisodeStats]) -> float | None:
    stuck_steps = sum(s.stuck_seconds * s.steps / max(s.seconds, 1e-9) for s in stats)
    if stuck_steps <= 0:
        return None
    return float(min(sum(s.stuck_limb_steps for s in stats) / stuck_steps, 1.0))


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
        # terrain / task metrics
        "ang_vel_rmse": float(np.mean([s.ang_vel_rmse for s in stats])),
        "centre_drift_m": float(np.mean([s.centre_drift_m for s in stats])),
        "climb_height_p50": float(np.median([s.climb_height for s in stats])),
        # distance / (|commanded vx| x seconds); undefined (None = "not evaluated") for a zero command
        "progress_ratio": (float(np.median(d) / (abs(stats[0].cmd[0]) * episode_seconds))
                           if abs(stats[0].cmd[0]) >= spec.ZERO_CMD_EPS else None),
        "stumble_rate": float(np.mean([s.stumble_rate for s in stats])),
        "foot_clearance_p50_mm": float(np.median([s.foot_clearance_p50_mm for s in stats])),
        "body_clearance_min_mm": float(np.min([s.body_clearance_min_mm for s in stats])),
        # stairs: median over episodes of the highest level reached, and of how far it descended again
        "climb_max_p50_m": float(np.median([s.climb_max_m for s in stats])),
        "descent_p50_m": float(np.median([s.descent_m for s in stats])),
        # servo safety: the WORST episode, not the mean — one bad episode is one broken servo
        "peak_joint_speed_rad_s": float(np.max([s.peak_joint_speed_rad_s for s in stats])),
        "max_joint_speed_rad_s": float(np.max([s.max_joint_speed_rad_s for s in stats])),
        "stall_fraction": float(np.max([s.stall_fraction for s in stats])),
        "stall_concurrent_max": int(np.max([s.stall_concurrent_max for s in stats])),
        "travel_deg_max": float(np.max([s.travel_deg_max for s in stats])),
        # stuck diagnostics: how long, and where (x of the first stall, median over the episodes that stalled)
        "stuck_seconds_p50": float(np.median([s.stuck_seconds for s in stats])),
        "stuck_seconds_max": float(np.max([s.stuck_seconds for s in stats])),
        "stuck_episode_rate": float(np.mean([s.stuck_x is not None for s in stats])),
        "stuck_x_p50": (float(np.median([s.stuck_x for s in stats if s.stuck_x is not None]))
                        if any(s.stuck_x is not None for s in stats) else None),
        # share of stalled steps with a shank/thigh on the ground: 1.0 = every stall is an edge catch
        "stuck_limb_share": _stuck_limb_share(stats),
    }


# ── protocols ──────────────────────────────────────────────────────────────

def nominal_env(cfg: TrainConfig, *, variant: str | None = None, envelope_tier: str = "agent",
                model_params: dr_mod.ModelParams | None = None,
                episode_params: dr_mod.EpisodeParams | None = None,
                terrain: tr.TerrainField | None = None, spawn_jitter_m: float | None = None) -> BittleCpuEnv:
    """Env with DR fixed to nominal (or the given overrides) — the benchmark protocol.

    ``terrain`` pins the ground (a suite protocol overrides whatever the run trained on) and
    picks the XML variant; without it the run's own terrain config is used.
    """
    if terrain is not None and variant is None:
        variant = "cpu_terrain" if terrain.n_boxes > 0 else "cpu"
    return BittleCpuEnv(cfg, variant=variant, seed=0, envelope_tier=envelope_tier,
                        dr_override=model_params or dr_mod.nominal_model_params(),
                        episode_override=episode_params or dr_mod.nominal_episode_params(),
                        terrain_override=terrain, spawn_jitter_m=spawn_jitter_m)


def protocol_kwargs(proto: dict[str, Any]) -> dict[str, Any]:
    """Env-level kwargs shared by every sub-protocol of a suite: its fixed terrain + spawn jitter,
    and a fixed push schedule when the protocol has one (statue_v1)."""
    pt = proto.get("terrain") or {"kind": "flat"}
    out = {"terrain": tr.field_from_protocol(pt), "spawn_jitter_m": float(pt.get("spawn_jitter_m", 0.0)),
           "variant": tr.variant_for(pt.get("kind", "flat"), "cpu")}
    push = proto.get("push")
    if push:
        ep = dr_mod.nominal_episode_params()
        ep.push_enabled, ep.push_vel, ep.push_interval_s = True, float(push["vel"]), float(push["interval_s"])
        out["episode_params"] = ep
    return out


DR_PRESETS: dict[str, dict[str, Any]] = {
    "friction_low": {"friction": 0.5},
    "friction_high": {"friction": 1.1},
    "payload_30g": {"payload_kg": 0.03},
    "kp_low": {"kp": 6.0},
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
                      command_vx: float | None = None, command: Any = None, variant: str | None = None,
                      envelope_tier: str = "agent", model_params=None, episode_params=None,
                      record_seeds: set[int] | None = None, source: dict[str, Any] | None = None,
                      terrain: tr.TerrainField | None = None, spawn_jitter_m: float | None = None,
                      ) -> tuple[list[EpisodeStats], dict[int, dict[str, Any]]]:
    env = nominal_env(cfg, variant=variant, envelope_tier=envelope_tier, model_params=model_params,
                      episode_params=episode_params, terrain=terrain, spawn_jitter_m=spawn_jitter_m)
    cmd = np.array(command, dtype=np.float64) if command is not None else np.array([command_vx or 0.0, 0.0, 0.0])
    variant = env.variant
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
