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
from ..env.gpu_env import PRIVILEGED_VERSION, BittleGpuEnv, make_domain_randomizer
from ..policy.export import export_policy
from .bar_eval import BarEval


def parent_privileged_version(parent_dir: Path) -> int:
    """The critic-input layout version the parent's policy.npz was exported with (1 = pre-terrain)."""
    npz = parent_dir / "policy" / "policy.npz"
    if not npz.is_file():
        return 1
    try:
        z = np.load(npz, allow_pickle=False)
        return int(json.loads(str(z["meta_json"])).get("privileged_version", 1))
    except Exception:  # pragma: no cover
        return 1


def load_restore_params(parent_dir: Path, cfg: TrainConfig) -> tuple[Any, str | None]:
    """Parent params for a warm start, or (None, reason) when the shapes cannot match.

    The value network's input (``privileged_state``) is versioned separately from the actor
    obs: a parent exported under an older layout would crash deep inside brax with a shape
    error, so it degrades to a clean cold start with the reason recorded instead.
    """
    path = parent_dir / "policy" / "params.pkl"
    if not path.is_file():
        return None, "parent has no params.pkl"
    try:
        pcfg = TrainConfig.model_validate(json.loads((parent_dir / "config.json").read_text()))
    except Exception as exc:  # pragma: no cover
        return None, f"parent config unreadable: {exc}"
    same = (tuple(pcfg.ppo.policy_hidden) == tuple(cfg.ppo.policy_hidden)
            and tuple(pcfg.ppo.value_hidden) == tuple(cfg.ppo.value_hidden)
            and pcfg.obs.history_n == cfg.obs.history_n and pcfg.obs.phase_clock == cfg.obs.phase_clock)
    if not same:
        return None, "network/observation shape differs from the parent"
    pv = parent_privileged_version(parent_dir)
    if pv != PRIVILEGED_VERSION:
        return None, f"critic input layout differs (parent privileged_version {pv}, this trainer {PRIVILEGED_VERSION}); cold start"
    with open(path, "rb") as fh:
        return pickle.load(fh), None


def train_policy(cfg: TrainConfig, run_dir: Path, *, progress: Callable[[dict[str, Any]], None] | None = None,
                 impl: str | None = None, sim_dt: float = 0.002, restore_params: Any = None) -> dict[str, Any]:
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
    # the bar eval: the policy on its suite's protocol terrain + command, next to every curve point
    try:
        bar = BarEval(cfg, impl=impl, sim_dt=sim_dt)
    except Exception as exc:  # a bar failure must never take the training run down
        print(f"bar eval disabled: {exc!r}")
        bar = None
    t0 = time.time()
    last = {"t": t0, "steps": 0}
    curve: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def emit(point: dict[str, Any]) -> None:
        if bar is not None and bar.latest:
            point.update(bar.latest)
        curve.append(point)
        if progress:
            progress(point)

    def policy_params_fn(current_step: int, make_policy, params) -> None:
        # brax calls this BEFORE the evaluation of the same params (and once right after progress_fn(0))
        if bar is not None and bar.enabled:
            try:
                bar(current_step, make_policy, params)
            except Exception as exc:  # pragma: no cover - GPU hiccup: keep training
                print(f"bar eval failed at step {current_step}: {exc!r}")
        while pending:
            emit(pending.pop(0))

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
        if num_steps == 0 and bar is not None and bar.enabled and not bar.latest:
            pending.append(point)  # the initial bar rollout follows this call; pair them
        else:
            emit(point)

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
        restore_params=restore_params,
        policy_params_fn=policy_params_fn,
    )
    make_inference_fn, params, metrics = train_fn(environment=env, eval_env=eval_env)
    elapsed = time.time() - t0
    while pending:
        emit(pending.pop(0))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "policy").mkdir(exist_ok=True)
    with open(run_dir / "policy" / "params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    export_policy(str(run_dir / "policy" / "policy.npz"), params, obs_layout=env.obs_layout, action_size=8,
                  config_hash=cfg.config_hash(), extra={"impl": env.mjx_model.impl.value, "sim_dt": sim_dt,
                                                       "privileged_version": PRIVILEGED_VERSION,
                                                       "task": cfg.task, "terrain_kind": cfg.terrain.kind})

    final = curve[-1] if curve else {}
    out = {
        "num_timesteps": p.num_timesteps,
        "num_envs": p.num_envs,
        "elapsed_s": round(elapsed, 1),
        "steps_per_s_mean": round(p.num_timesteps / max(elapsed, 1e-6), 1),
        "reward_first": curve[0]["reward"] if curve else None,
        "reward_final": final.get("reward"),
        "distance_final": final.get("distance_x"),
        # the policy on its suite's protocol (terrain + command) -- comparable to the benchmark, unlike distance_final
        "bar_distance_final": final.get("bar_distance_x"),
        "bar_distance_p50_final": final.get("bar_distance_p50"),
        "bar_fall_rate_final": final.get("bar_fall_rate"),
        "bar_protocol": bar.protocol if bar is not None else None,
        "episode_length_final": final.get("episode_length"),
        "impl": env.mjx_model.impl.value,
        "sim_dt": sim_dt,
        "warm_start": restore_params is not None,
        "curve": curve,
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
