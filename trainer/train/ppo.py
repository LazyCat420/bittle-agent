"""Brax PPO on the GPU env (mujoco_playground pattern)."""

from __future__ import annotations

import functools
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

from . import compat  # noqa: F401  (must precede brax imports)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from ..config import TrainConfig
from ..env.gpu_env import BittleGpuEnv, make_domain_randomizer
from ..policy.export import export_policy


def train_policy(cfg: TrainConfig, run_dir: Path, *, progress: Callable[[dict[str, Any]], None] | None = None,
                 impl: str | None = None, sim_dt: float = 0.002) -> dict[str, Any]:
    """Train and export. Returns training metrics (also written to run_dir/metrics.json)."""
    cfg = cfg.resolved()
    p = cfg.ppo
    env = BittleGpuEnv(cfg, impl=impl, num_envs=p.num_envs, sim_dt=sim_dt)
    eval_env = BittleGpuEnv(cfg, impl=impl, num_envs=128, sim_dt=sim_dt)
    rand_fn = make_domain_randomizer(cfg, env.mj_model)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(p.policy_hidden),
        value_hidden_layer_sizes=tuple(p.value_hidden),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    t0 = time.time()
    last = {"t": t0, "steps": 0}
    curve: list[dict[str, Any]] = []

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        now = time.time()
        sps = (num_steps - last["steps"]) / max(now - last["t"], 1e-6)
        last["t"], last["steps"] = now, num_steps
        point = {
            "step": int(num_steps),
            "reward": float(metrics.get("eval/episode_reward", float("nan"))),
            "episode_length": float(metrics.get("eval/avg_episode_length", float("nan"))),
            "elapsed_s": round(now - t0, 1),
            "steps_per_s": round(sps, 1),
            "distance_x": float(metrics.get("eval/episode_distance_x", float("nan"))),
        }
        for k, v in metrics.items():
            if k.startswith("eval/episode_reward/"):
                point[k.replace("eval/episode_", "")] = float(v)
        curve.append(point)
        if progress:
            progress(point)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=p.num_timesteps,
        num_evals=p.num_evals,
        episode_length=cfg.episode_steps,
        action_repeat=p.action_repeat,
        unroll_length=p.unroll_length,
        num_minibatches=p.num_minibatches,
        num_updates_per_batch=p.num_updates_per_batch,
        discounting=p.discounting,
        learning_rate=p.learning_rate,
        entropy_cost=p.entropy_cost,
        num_envs=p.num_envs,
        batch_size=p.batch_size,
        reward_scaling=p.reward_scaling,
        normalize_observations=p.normalize_observations,
        clipping_epsilon=p.clipping_epsilon,
        seed=cfg.seed,
        network_factory=network_factory,
        randomization_fn=rand_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress_fn,
        num_eval_envs=128,
    )
    make_inference_fn, params, metrics = train_fn(environment=env, eval_env=eval_env)
    elapsed = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "policy").mkdir(exist_ok=True)
    with open(run_dir / "policy" / "params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    export_policy(str(run_dir / "policy" / "policy.npz"), params, obs_layout=env.obs_layout, action_size=8,
                  config_hash=cfg.config_hash(), extra={"impl": env.mjx_model.impl.value, "sim_dt": sim_dt})

    final = curve[-1] if curve else {}
    out = {
        "num_timesteps": p.num_timesteps,
        "num_envs": p.num_envs,
        "elapsed_s": round(elapsed, 1),
        "steps_per_s_mean": round(p.num_timesteps / max(elapsed, 1e-6), 1),
        "reward_first": curve[0]["reward"] if curve else None,
        "reward_final": final.get("reward"),
        "distance_final": final.get("distance_x"),
        "episode_length_final": final.get("episode_length"),
        "impl": env.mjx_model.impl.value,
        "sim_dt": sim_dt,
        "curve": curve,
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
