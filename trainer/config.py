"""TrainConfig: the ONLY thing the LLM is allowed to edit.

Every field is bounded, unknown keys are rejected (``extra="forbid"``), and
``config_hash()`` gives a stable identity for a resolved config so runs are
reproducible and comparable. The reward is parameterised by weights; no code
is ever generated or executed from a config.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA_VERSION = "1"

Range = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_range(v: Range, lo: float, hi: float, name: str) -> Range:
    a, b = float(v[0]), float(v[1])
    if a > b:
        raise ValueError(f"{name}: lower bound {a} > upper bound {b}")
    if a < lo or b > hi:
        raise ValueError(f"{name}: [{a}, {b}] outside allowed [{lo}, {hi}]")
    return (a, b)


class RewardWeights(_Strict):
    """Per-step reward weights. Positive = reward, negative = penalty."""

    tracking_lin_vel: float = Field(1.5, ge=0.0, le=10.0)
    tracking_ang_vel: float = Field(0.5, ge=0.0, le=10.0)
    lin_vel_z: float = Field(-1.0, ge=-10.0, le=0.0)
    ang_vel_xy: float = Field(-0.05, ge=-10.0, le=0.0)
    orientation: float = Field(-2.0, ge=-10.0, le=0.0)
    base_height: float = Field(-1.0, ge=-10.0, le=0.0)
    action_rate: float = Field(-0.01, ge=-10.0, le=0.0)
    energy: float = Field(-0.0005, ge=-10.0, le=0.0)
    joint_saturation: float = Field(-1.0, ge=-10.0, le=0.0)
    feet_air_time: float = Field(0.1, ge=0.0, le=10.0)
    feet_slip: float = Field(-0.05, ge=-10.0, le=0.0)
    stand_still: float = Field(-0.2, ge=-10.0, le=0.0)


class RewardConfig(_Strict):
    weights: RewardWeights = Field(default_factory=RewardWeights)
    #: exp(-err^2 / sigma). Bittle moves at 0.1-0.2 m/s, ~10x tighter than Go1.
    tracking_sigma: float = Field(0.01, ge=0.0005, le=1.0)
    ang_tracking_sigma: float = Field(0.25, ge=0.01, le=5.0)
    #: settled torso height in the stand pose (m); recomputed by build_models.
    base_height_target: float = Field(0.048, ge=0.02, le=0.09)


class CommandConfig(_Strict):
    """Velocity command sampling ranges (m/s, rad/s)."""

    vx: Range = (0.05, 0.20)
    vy: Range = (0.0, 0.0)
    wz: Range = (0.0, 0.0)
    resample_s: float = Field(5.0, ge=0.5, le=30.0)

    @field_validator("vx")
    @classmethod
    def _vx(cls, v: Range) -> Range:
        return _check_range(v, -0.3, 0.4, "commands.vx")

    @field_validator("vy")
    @classmethod
    def _vy(cls, v: Range) -> Range:
        return _check_range(v, -0.15, 0.15, "commands.vy")

    @field_validator("wz")
    @classmethod
    def _wz(cls, v: Range) -> Range:
        return _check_range(v, -1.5, 1.5, "commands.wz")


class DRConfig(_Strict):
    """Domain randomisation ranges, sampled per episode."""

    enabled: bool = True
    friction: Range = (0.5, 1.1)
    mass_scale: Range = (0.85, 1.15)
    payload_g: Range = (0.0, 30.0)
    com_shift_mm: Range = (-10.0, 10.0)
    kp: Range = (6.0, 15.0)
    forcerange: Range = (0.15, 0.35)
    damping: Range = (0.02, 0.10)
    frictionloss: Range = (0.0, 0.03)
    latency_steps: tuple[int, int] = (0, 3)
    gyro_noise: float = Field(0.1, ge=0.0, le=1.0)
    gyro_bias: float = Field(0.05, ge=0.0, le=0.5)
    gravity_noise: float = Field(0.05, ge=0.0, le=0.5)
    init_joint_noise_deg: float = Field(5.0, ge=0.0, le=30.0)
    init_tilt_rad: float = Field(0.02, ge=0.0, le=0.3)
    push_enabled: bool = False
    push_vel: Range = (0.1, 0.3)
    push_interval_s: Range = (3.0, 6.0)

    @field_validator("friction")
    @classmethod
    def _f(cls, v: Range) -> Range:
        return _check_range(v, 0.2, 2.0, "dr.friction")

    @field_validator("mass_scale")
    @classmethod
    def _m(cls, v: Range) -> Range:
        return _check_range(v, 0.5, 1.5, "dr.mass_scale")

    @field_validator("payload_g")
    @classmethod
    def _p(cls, v: Range) -> Range:
        return _check_range(v, 0.0, 100.0, "dr.payload_g")

    @field_validator("com_shift_mm")
    @classmethod
    def _c(cls, v: Range) -> Range:
        return _check_range(v, -30.0, 30.0, "dr.com_shift_mm")

    @field_validator("kp")
    @classmethod
    def _kp(cls, v: Range) -> Range:
        return _check_range(v, 2.0, 60.0, "dr.kp")

    @field_validator("forcerange")
    @classmethod
    def _fr(cls, v: Range) -> Range:
        return _check_range(v, 0.05, 1.0, "dr.forcerange")

    @field_validator("damping")
    @classmethod
    def _d(cls, v: Range) -> Range:
        return _check_range(v, 0.005, 2.0, "dr.damping")

    @field_validator("frictionloss")
    @classmethod
    def _fl(cls, v: Range) -> Range:
        return _check_range(v, 0.0, 0.5, "dr.frictionloss")

    @field_validator("latency_steps")
    @classmethod
    def _lat(cls, v: tuple[int, int]) -> tuple[int, int]:
        a, b = int(v[0]), int(v[1])
        if not (0 <= a <= b <= 6):
            raise ValueError("dr.latency_steps must satisfy 0 <= lo <= hi <= 6")
        return (a, b)

    @field_validator("push_vel")
    @classmethod
    def _pv(cls, v: Range) -> Range:
        return _check_range(v, 0.0, 1.0, "dr.push_vel")

    @field_validator("push_interval_s")
    @classmethod
    def _pi(cls, v: Range) -> Range:
        return _check_range(v, 0.5, 30.0, "dr.push_interval_s")


class PPOConfig(_Strict):
    num_envs: int = Field(2048, ge=1, le=8192)
    num_timesteps: int = Field(40_000_000, ge=1_000, le=200_000_000)
    unroll_length: int = Field(20, ge=1, le=100)
    num_minibatches: int = Field(32, ge=1, le=256)
    num_updates_per_batch: int = Field(4, ge=1, le=32)
    discounting: float = Field(0.97, ge=0.8, le=0.999)
    learning_rate: float = Field(3e-4, ge=1e-6, le=1e-2)
    entropy_cost: float = Field(1e-2, ge=0.0, le=0.1)
    batch_size: int = Field(256, ge=8, le=8192)
    policy_hidden: tuple[int, ...] = (256, 256)
    value_hidden: tuple[int, ...] = (256, 256)
    num_evals: int = Field(10, ge=1, le=100)
    reward_scaling: float = Field(1.0, ge=0.01, le=100.0)
    normalize_observations: bool = True
    clipping_epsilon: float = Field(0.2, ge=0.01, le=0.5)
    action_repeat: int = Field(1, ge=1, le=4)

    @field_validator("policy_hidden", "value_hidden")
    @classmethod
    def _hidden(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        v = tuple(int(x) for x in v)
        if not (1 <= len(v) <= 4) or any(not (8 <= x <= 1024) for x in v):
            raise ValueError("hidden layers: 1-4 layers of 8-1024 units")
        return v


class ObsConfig(_Strict):
    history_n: int = Field(3, ge=1, le=10)
    phase_clock: bool = False


class TrainConfig(_Strict):
    schema_version: Literal["1"] = SCHEMA_VERSION
    reward: RewardConfig = Field(default_factory=RewardConfig)
    #: 0 = forward only; 1 = + turning; 2 = + lateral + pushes.
    curriculum_stage: int = Field(0, ge=0, le=2)
    commands: CommandConfig = Field(default_factory=CommandConfig)
    dr: DRConfig = Field(default_factory=DRConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    episode_seconds: float = Field(10.0, ge=2.0, le=60.0)
    control_hz: Literal[25, 50] = 50
    action_scale_deg: float = Field(6.0, ge=1.0, le=20.0)
    action_mode: Literal["delta", "residual_from_stand"] = "delta"
    obs: ObsConfig = Field(default_factory=ObsConfig)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    sim_impl: Literal["warp", "jax"] = "warp"
    #: Start from the parent run's policy + normaliser when base_run_id is given (same network
    #: shape). A reward tweak then needs ~10M steps, not 40M from scratch.
    init_from_parent: bool = True
    notes: str = Field("", max_length=2000)

    # ── derived helpers ────────────────────────────────────────────────
    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def episode_steps(self) -> int:
        return int(round(self.episode_seconds * self.control_hz))

    def resolved(self) -> "TrainConfig":
        """Configs are fully explicit after ``apply_patch`` (stage defaults are materialised
        there, where explicit patch paths are known). Kept for call-site symmetry."""
        return self

    def config_hash(self) -> str:
        canon = json.dumps(self.resolved().model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ── curriculum ────────────────────────────────────────────────────────────

#: Stage-managed fields and their per-stage defaults. A field keeps a stage
#: default only until the caller patches it explicitly.
STAGE_DEFAULTS: dict[int, dict[str, Any]] = {
    0: {"commands.vx": (0.05, 0.20), "commands.vy": (0.0, 0.0), "commands.wz": (0.0, 0.0), "dr.push_enabled": False},
    1: {"commands.vx": (0.0, 0.25), "commands.vy": (0.0, 0.0), "commands.wz": (-0.5, 0.5), "dr.push_enabled": False},
    2: {"commands.vx": (0.0, 0.25), "commands.vy": (-0.05, 0.05), "commands.wz": (-0.5, 0.5), "dr.push_enabled": True},
}


def _get_path(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _norm(v: Any) -> Any:
    return tuple(v) if isinstance(v, (list, tuple)) else v


def _patch_paths(patch: dict[str, Any], prefix: str = "") -> set[str]:
    out: set[str] = set()
    for k, v in patch.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out |= _patch_paths(v, path)
        else:
            out.add(path)
    return out


def apply_stage_defaults(merged: dict[str, Any], *, base_stage: int, explicit: set[str]) -> dict[str, Any]:
    """Move stage-managed fields from the base stage's defaults to the new stage's."""
    new_stage = int(merged.get("curriculum_stage", 0))
    for path, new_default in STAGE_DEFAULTS[new_stage].items():
        if path in explicit:
            continue
        cur = _norm(_get_path(merged, path))
        was_default = cur is None or cur == _norm(STAGE_DEFAULTS[base_stage][path]) or cur == _norm(STAGE_DEFAULTS[0][path])
        if was_default:
            _set_path(merged, path, new_default)
    return merged


# ── patching / diffing ────────────────────────────────────────────────────


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_patch(base: TrainConfig | dict[str, Any] | None, patch: dict[str, Any] | None) -> TrainConfig:
    """Validate ``base`` + ``patch`` and materialise curriculum-stage defaults.

    Raises ``pydantic.ValidationError``. Fields the patch names explicitly
    always win; stage-managed fields still at a stage default follow the new stage.
    """
    base_d = base.model_dump(mode="json") if isinstance(base, TrainConfig) else dict(base or {})
    base_stage = int(base_d.get("curriculum_stage", 0))
    patch = patch or {}
    merged = deep_merge(base_d, patch)
    merged = apply_stage_defaults(merged, base_stage=base_stage, explicit=_patch_paths(patch))
    return TrainConfig.model_validate(merged)


def config_diff(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Flat list of {path, from, to} for leaves that differ."""
    out: list[dict[str, Any]] = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        path = f"{prefix}.{k}" if prefix else k
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(config_diff(va, vb, path))
        elif va != vb:
            out.append({"path": path, "from": va, "to": vb})
    return out


def validation_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]


def default_config() -> TrainConfig:
    return TrainConfig()


def json_schema() -> dict[str, Any]:
    return TrainConfig.model_json_schema()
