"""Benchmark subprocess: ``python -m trainer.eval.benchmark_cli --run-id <id> [--dr-sweep] [--dual-sim]``.

Reads the run's config + policy.npz, runs the gate protocol in CPU MuJoCo on
the full-mesh model, optionally the DR sweep and the GPU-vs-CPU consistency
check, writes ``benchmark/<suite>@<version>/report.json`` + rollouts, and
marks the run ``done``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from ..config import TrainConfig
from ..policy.mlp import NumpyPolicy
from ..store.runs import RunStore
from .evaluator import DR_PRESETS, PolicyController, aggregate, evaluate_protocol, preset_params
from .gates import evaluate_gates, load_suite

REPO = Path(__file__).resolve().parent.parent.parent


def benchmark_run(store: RunStore, run_id: str, *, suite_name: str = "flat_v1", n_episodes: int | None = None,
                  dr_sweep: bool = False, dual_sim: bool = False, record_n: int = 3) -> dict:
    cfg = TrainConfig.model_validate(store.config(run_id)).resolved()
    suite = load_suite(suite_name)
    proto = suite["protocol"]
    n = int(n_episodes or proto["n_episodes"])
    seed0 = int(proto["seed_start"])
    seconds = float(proto["episode_seconds"])
    vx = float(proto["command_vx"])
    policy = NumpyPolicy.load(store.policy_path(run_id))
    ctrl = PolicyController(policy)
    groups: set[str] = set()

    # 1. nominal protocol
    stats, rollouts = evaluate_protocol(cfg, ctrl, n_episodes=n, seed_start=seed0, seconds=seconds, command_vx=vx,
                                        record_seeds=set(range(seed0, seed0 + record_n)),
                                        source={"run_id": run_id, "suite": suite_name})
    metrics = aggregate(stats, seconds)

    # 2. stand still (5 s, zero command)
    ss, _ = evaluate_protocol(cfg, ctrl, n_episodes=max(3, n // 4), seed_start=seed0 + 100, seconds=5.0, command_vx=0.0)
    metrics["stand_still_drift_m"] = float(np.mean([abs(s.distance_x) + abs(s.lateral_y) for s in ss]))
    metrics["stand_still_falls"] = int(sum(1 for s in ss if s.fell))

    # 3. multi command (stage >= 1)
    if cfg.curriculum_stage >= 1:
        rmses = []
        for v in (0.05, 0.10, 0.18):
            st, _ = evaluate_protocol(cfg, ctrl, n_episodes=max(3, n // 4), seed_start=seed0 + 200, seconds=seconds, command_vx=v)
            rmses.append(aggregate(st, seconds)["vel_tracking_rmse"])
        metrics["multi_command_rmse_max"] = float(max(rmses))
        metrics["multi_command_rmse"] = rmses

    # 4. DR sweep
    if dr_sweep:
        groups.add("dr_sweep")
        sweep = {}
        for name in DR_PRESETS:
            mp, ep = preset_params(name)
            st, _ = evaluate_protocol(cfg, ctrl, n_episodes=max(5, n // 2), seed_start=seed0 + 300, seconds=seconds,
                                      command_vx=vx, model_params=mp, episode_params=ep)
            agg = aggregate(st, seconds)
            sweep[name] = {"fall_rate": agg["fall_rate"], "distance_p50": agg["forward_distance_p50"]}
        metrics["dr_sweep"] = sweep
        metrics["dr_fall_rate_max"] = float(max(v["fall_rate"] for v in sweep.values()))
        nominal_d = max(metrics["forward_distance_p50"], 1e-6)
        metrics["dr_distance_ratio_min"] = float(min(v["distance_p50"] for v in sweep.values()) / nominal_d)

    # 5. baseline comparison (cached trot)
    context: dict = {}
    base = store.baselines().get("opencat_trF")
    if base and base.get("metrics"):
        groups.add("baseline")
        bd = max(float(base["metrics"].get("forward_distance_p50", 0.0)), 1e-6)
        metrics["baseline_distance_ratio"] = float(metrics["forward_distance_p50"] / bd)
        metrics["baseline_fall_delta"] = float(metrics["fall_rate"] - float(base["metrics"].get("fall_rate", 0.0)))
        be = float(base["metrics"].get("energy_proxy_w", 0.0))
        if be > 0:
            metrics["baseline_energy_ratio"] = float(metrics["energy_proxy_w"] / be)
        context["baseline_metrics"] = base["metrics"]
        context["baseline_name"] = "opencat_trF"

    # 5b. context for the reflection: parent run's gate metrics + reward-term breakdown of the last eval
    parent = store.state(run_id).get("parent")
    if parent and store.exists(parent):
        prep = store.benchmark(parent, suite_name)
        if prep and prep.get("metrics"):
            context["parent_metrics"] = prep["metrics"]
            context["parent_run_id"] = parent
    curve = store.curve(run_id, limit=1000)
    if curve:
        last = curve[-1]
        context["reward_breakdown"] = {k.replace("reward/", ""): float(v) for k, v in last.items()
                                       if k.startswith("reward/") and not k.endswith("_std")}

    # 6. dual-sim consistency (GPU engine vs CPU, shared seeds)
    if dual_sim:
        try:
            from .dual_sim import gpu_distances

            groups.add("dual_sim")
            gpu_d = gpu_distances(cfg, policy, seeds=list(range(seed0, seed0 + 5)), seconds=seconds, command_vx=vx)
            cpu_d = np.median([s.distance_x for s in stats[:5]])
            metrics["dual_sim_gpu_distance_p50"] = float(np.median(gpu_d))
            metrics["dual_sim_distance_ratio"] = float(np.median(gpu_d) / max(cpu_d, 1e-6))
        except Exception as exc:  # GPU unavailable in this process: report, don't fail the benchmark
            metrics["dual_sim_error"] = str(exc)[:300]

    report = evaluate_gates(metrics, suite, curriculum_stage=cfg.curriculum_stage, groups_enabled=groups, context=context)
    report.update({"run_id": run_id, "n_episodes": n, "seeds": [s.seed for s in stats],
                   "sim": f"mujoco-cpu/{proto['model']}", "episodes": [s.to_dict() for s in stats],
                   "config_hash": cfg.config_hash()})
    store.write_benchmark(run_id, suite["suite"], str(suite["version"]), report)
    for seed, ro in rollouts.items():
        p = store.rollout_path(run_id, suite["suite"], str(suite["version"]), seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ro))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suite", default="flat_v1")
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--dr-sweep", action="store_true")
    ap.add_argument("--dual-sim", action="store_true")
    ap.add_argument("--runs-dir", default=os.getenv("TRAINER_RUNS_DIR", str(REPO / "runs")))
    a = ap.parse_args(argv)
    store = RunStore(a.runs_dir)
    try:
        report = benchmark_run(store, a.run_id, suite_name=a.suite, n_episodes=a.n_episodes,
                               dr_sweep=a.dr_sweep, dual_sim=a.dual_sim)
        store.update_state(a.run_id, status="done")
        print(report["reflection"])
        return 0
    except Exception as exc:
        traceback.print_exc()
        store.update_state(a.run_id, status="trained", error=f"benchmark failed: {exc}"[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
