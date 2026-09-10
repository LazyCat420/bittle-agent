"""The "bar" eval: the policy rolled out on its task's BENCHMARK protocol during training.

Brax's own eval curve (``distance_x``) runs on the training distribution with random commands, so it
reads healthy while the benchmark fails (rough r9/r10, 2026-09-07: 0.9 m in training, 0.3 m on the
suite). ``BarEval`` rolls the current params out on the suite protocol's terrain (every box at the
protocol height, nominal DR) with the protocol command, and reports ``bar_distance_x`` /
``bar_fall_rate`` next to every curve point, so a train/benchmark mismatch shows up while the run is
still training instead of after it.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jp

from ..config import TrainConfig, apply_patch
from ..env import terrain as tr
from ..env.gpu_env import BittleGpuEnv, make_domain_randomizer
from ..tasks import TASKS


def bar_protocol(cfg: TrainConfig) -> tuple[TrainConfig, tuple[float, float, float]] | None:
    """(config whose terrain is the suite protocol's, protocol command) -- or None when the task's suite
    has no single fixed protocol (scenes suites judge nine terrains; they get no bar curve)."""
    from ..eval.gates import load_suite  # local: gates imports the evaluator stack

    task = TASKS.get(cfg.task)
    if task is None:
        return None
    suite = load_suite(task.suite)
    if suite.get("scenes"):
        return None
    proto = suite["protocol"]
    pt = dict(proto.get("terrain") or {"kind": "flat"})
    kind = pt.get("kind", "flat")
    if kind not in tr.KINDS:
        return None
    terrain: dict[str, Any] = {"kind": kind, "slope_share": 1.0, "box_share": 1.0}
    if tr.has_slope(kind):
        terrain["slope_deg"] = [float(pt.get("slope_deg", 0.0))] * 2
        terrain["slope_yaw_deg"] = [float(pt.get("slope_yaw_deg", 0.0))] * 2
    if tr.has_boxes(kind):
        terrain.update({"n_boxes": int(pt.get("n_boxes", 0)),
                        "box_height_m": [float(pt["box_height_m"])] * 2,
                        "box_size_m": [float(pt.get("box_size_m", 0.035))] * 2,
                        "box_spacing_m": float(pt.get("box_spacing_m", 0.10)),
                        "field_start_m": float(pt.get("field_start_m", 0.15)),
                        "field_width_m": float(pt.get("field_width_m", 0.40)),
                        "spawn_jitter_m": float(pt.get("spawn_jitter_m", 0.0))})
    else:
        terrain.update({"n_boxes": 0, "spawn_jitter_m": 0.0})
    # the benchmark runs nominal DR: no pushes, no payload, no latency draw
    bar_cfg = apply_patch(cfg, {"terrain": terrain, "dr": {"enabled": False, "push_enabled": False}})
    cmd = tuple(float(x) for x in proto["command"])
    return bar_cfg, cmd


class BarEval:
    """Rolls the current policy out on the bar protocol; ``latest`` holds the last result."""

    def __init__(self, cfg: TrainConfig, *, impl: str | None, sim_dt: float, num_envs: int = 128, seed: int = 0):
        from mujoco_playground import wrapper

        self.latest: dict[str, float] = {}
        self.protocol: dict[str, Any] | None = None
        spec = bar_protocol(cfg)
        self.enabled = spec is not None
        if spec is None:
            return
        bar_cfg, cmd = spec
        self.protocol = {"command": list(cmd), "terrain": bar_cfg.terrain.model_dump(mode="json"), "num_envs": num_envs}
        env = BittleGpuEnv(bar_cfg, impl=impl, num_envs=num_envs, sim_dt=sim_dt, fixed_command=cmd)
        rand = make_domain_randomizer(bar_cfg, env.mj_model)
        keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self._env = wrapper.wrap_for_brax_training(
            env, episode_length=cfg.episode_steps, action_repeat=cfg.ppo.action_repeat,
            randomization_fn=None if rand is None else functools.partial(rand, rng=keys))
        self._num_envs = num_envs
        self._steps = int(cfg.episode_steps)
        self._roll = None

    def _rollout(self, make_policy, params, rng):
        policy = make_policy(params, deterministic=True)
        rng, k_reset = jax.random.split(rng)
        state = self._env.reset(jax.random.split(k_reset, self._num_envs))
        alive = jp.ones(self._num_envs)
        fell = jp.zeros(self._num_envs)
        dist = jp.zeros(self._num_envs)

        def body(carry, t):
            state, alive, fell, dist, key = carry
            key, k = jax.random.split(key)
            act, _ = policy(state.obs, k)
            state = self._env.step(state, act)
            dist = dist + state.metrics["distance_x"] * alive
            done = state.done
            fell = jp.maximum(fell, alive * done * (t < self._steps - 1))
            alive = alive * (1.0 - done)
            return (state, alive, fell, dist, key), None

        (_, _, fell, dist, _), _ = jax.lax.scan(body, (state, alive, fell, dist, rng), jp.arange(self._steps))
        return dist, fell

    def __call__(self, current_step: int, make_policy, params) -> None:
        """brax ``policy_params_fn``: runs the bar rollout on the params just evaluated."""
        if not self.enabled:
            return
        if self._roll is None:
            self._roll = jax.jit(functools.partial(self._rollout, make_policy))
        dist, fell = self._roll(params, jax.random.PRNGKey(int(current_step) + 1))
        dist = jax.device_get(dist)
        self.latest = {"bar_distance_x": float(dist.mean()), "bar_distance_p50": float(jp.median(dist)),
                       "bar_fall_rate": float(jax.device_get(fell).mean())}
