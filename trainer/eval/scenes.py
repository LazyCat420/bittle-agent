"""Run a suite's scenes (one or many protocols) and fold them into ONE metrics dict.

A classic suite is a single unnamed scene: its metrics keep their bare names and nothing
about the flat/slope/rough/... benchmarks changes. A ``scenes:`` suite (house_v1) runs every
scene and publishes:

* ``<scene>/<metric>`` for every aggregate metric of every scene (plus the scene's
  stand-still sub-protocol when requested);
* the bare top-level metrics = ``aggregate`` over the episodes of every FORWARD-walking
  scene (commanded vx > 0), which is what the baseline ratio gates and the score read;
* the servo-safety maxima (peak joint speed, stalls, saturation, smoothness, energy) taken
  over the WORST episode of ANY scene — one bad scene is one broken servo;
* ``fall_rate_max`` / ``progress_ratio_min`` over the scenes, ``stand_still_falls`` summed and
  ``stand_still_drift_m`` maxed over the scenes' stand-still sub-protocols.

Seeds: each scene's ``seed_start`` must differ (house_v1 spaces them by 1000) so recorded
rollouts never collide on ``seed_N.json`` and the viewer can replay any scene by seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import TrainConfig
from .evaluator import EpisodeStats, aggregate, evaluate_protocol, protocol_kwargs
from .gates import suite_scenes

#: aggregate keys where the suite-wide value is the WORST scene, not the forward-scene pool
WORST_OVER_SCENES = ("peak_joint_speed_rad_s", "max_joint_speed_rad_s", "stall_fraction", "stall_concurrent_max",
                     "travel_deg_max", "joint_saturation_pct", "action_smoothness_deg", "energy_proxy_w")


def run_suite_scenes(cfg: TrainConfig, controller, suite: dict[str, Any], *, n_episodes: int | None = None,
                     envelope_tier: str = "agent", record_n: int = 0, source: dict[str, Any] | None = None,
                     stand_still: bool = False) -> tuple[dict[str, Any], list[EpisodeStats], dict[int, dict[str, Any]], dict[str, Any]]:
    """Returns (metrics, all episode stats, rollouts by seed, primary protocol with ``scenes``)."""
    scenes = suite_scenes(suite)
    multi = len(scenes) > 1 or scenes[0][0] != ""
    metrics: dict[str, Any] = {}
    all_stats: list[EpisodeStats] = []
    forward_stats: list[EpisodeStats] = []
    rollouts: dict[int, dict[str, Any]] = {}
    per_scene: dict[str, dict[str, Any]] = {}
    seeds_used: set[int] = set()
    primary: dict[str, Any] | None = None
    ss_falls, ss_drift = 0, 0.0
    for name, proto in scenes:
        n = int(n_episodes or proto["n_episodes"])
        seed0 = int(proto["seed_start"])
        seconds = float(proto["episode_seconds"])
        cmd = [float(x) for x in proto["command"]]
        if multi and seed0 in seeds_used:
            raise ValueError(f"suite {suite.get('suite')!r}: scene {name!r} reuses seed_start {seed0}")
        seeds_used.add(seed0)
        common = protocol_kwargs(proto)
        src = dict(source or {}, terrain=proto["terrain"], command=cmd)
        if name:
            src["scene"] = name
        stats, ro = evaluate_protocol(cfg, controller, n_episodes=n, seed_start=seed0, seconds=seconds, command=cmd,
                                      envelope_tier=envelope_tier,
                                      record_seeds=set(range(seed0, seed0 + record_n)) if record_n else None,
                                      source=src, **common)
        for st in stats:
            st.scene = name
        agg = aggregate(stats, seconds)
        if stand_still:
            ss, _ = evaluate_protocol(cfg, controller, n_episodes=max(3, n // 4), seed_start=seed0 + 100, seconds=5.0,
                                      command=[0.0, 0.0, 0.0], envelope_tier=envelope_tier, **common)
            agg["stand_still_drift_m"] = float(np.mean([abs(s.distance_x) + abs(s.lateral_y) for s in ss]))
            agg["stand_still_falls"] = int(sum(1 for s in ss if s.fell))
            ss_falls += agg["stand_still_falls"]
            ss_drift = max(ss_drift, agg["stand_still_drift_m"])
        all_stats.extend(stats)
        rollouts.update(ro)
        if primary is None:
            primary = dict(proto, n_episodes=n)
        if not multi:
            metrics.update(agg)
            break
        per_scene[name] = agg
        for k, v in agg.items():
            metrics[f"{name}/{k}"] = v
        if cmd[0] > 0:
            forward_stats.extend(stats)
    if multi:
        pool = forward_stats or all_stats
        seconds = float(primary["episode_seconds"])
        top = aggregate(pool, seconds)
        # progress_ratio is per-command; pooled forward scenes have different speeds, so use per-episode
        top["progress_ratio"] = float(np.median([s.distance_x / max(abs(s.cmd[0]) * seconds, 1e-6) for s in pool]))
        for k in WORST_OVER_SCENES:
            top[k] = max(a[k] for a in per_scene.values())
        top["fall_rate_max"] = float(max(a["fall_rate"] for a in per_scene.values()))
        # progress along the COMMANDED direction (a backward scene's ratio is negative in aggregate)
        moving = [a["progress_ratio"] * np.sign(per_scene_cmd(scenes, n_)[0]) for n_, a in per_scene.items()
                  if a.get("progress_ratio") is not None]
        top["progress_ratio_min"] = float(min(moving)) if moving else 0.0
        top["n_scenes"] = len(per_scene)
        top["scenes"] = list(per_scene)
        if stand_still:
            top["stand_still_falls"] = ss_falls
            top["stand_still_drift_m"] = ss_drift
        metrics.update(top)
        primary["scenes"] = [n for n, _ in scenes]
        primary["scene"] = scenes[0][0]
    return metrics, all_stats, rollouts, primary or {}


def per_scene_cmd(scenes: list[tuple[str, dict[str, Any]]], name: str) -> list[float]:
    for n, proto in scenes:
        if n == name:
            return [float(x) for x in proto["command"]]
    return [0.0, 0.0, 0.0]
