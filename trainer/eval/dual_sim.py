"""GPU-engine rollouts of a numpy policy for the dual-sim consistency gate."""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

from ..config import TrainConfig
from ..env.gpu_env import BittleGpuEnv


def gpu_distances(cfg: TrainConfig, policy, *, seeds: list[int], seconds: float, command_vx: float,
                  impl: str | None = None, terrain=None, spawn_jitter_m: float | None = None) -> list[float]:
    """``terrain`` (a ``TerrainField`` from ``field_from_protocol``) pins the GPU model to the benchmark's
    fixed field so rough/slope protocols get the same dual-sim check as flat ones (before 2026-09-10 the
    GPU replay always ran on the parked, i.e. flat, model)."""
    cfg = cfg.model_copy(update={"dr": cfg.dr.model_copy(update={"enabled": False})})
    if spawn_jitter_m is not None:
        cfg = cfg.model_copy(update={"terrain": cfg.terrain.model_copy(update={"spawn_jitter_m": float(spawn_jitter_m)})})
    env = BittleGpuEnv(cfg, impl=impl, num_envs=1)
    if terrain is not None:
        env.pin_terrain(terrain)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    steps = int(round(seconds * cfg.control_hz))
    out = []
    for seed in seeds:
        st = reset(jax.random.PRNGKey(seed))
        st.info["command"] = jp.array([command_vx, 0.0, 0.0])
        st.info["steps_until_cmd"] = jp.int32(10 ** 6)
        x0 = float(st.data.qpos[0])
        for _ in range(steps):
            a = policy(np.asarray(st.obs["state"]))
            st = step(st, jp.asarray(a, dtype=jp.float32))
            st.info["command"] = jp.array([command_vx, 0.0, 0.0])
            if float(st.done) > 0:
                break
        out.append(float(st.data.qpos[0]) - x0)
    return out
