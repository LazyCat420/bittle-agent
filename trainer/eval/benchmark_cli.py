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
from ..tasks import TASKS
from .evaluator import DR_PRESETS, PolicyController, aggregate, evaluate_protocol, preset_params, protocol_kwargs
from .gates import evaluate_gates, load_suite, suite_protocol

REPO = Path(__file__).resolve().parent.parent.parent

#: A firmware-gait baseline that travels less than this (or falls in >= 90% of episodes) on a
#: protocol is DEGENERATE: its distance ratio would hand out a free pass, so the ratio gates are
#: reported as not evaluated instead. The absolute distance gate then carries the load.
MIN_BASELINE_DISTANCE_M = 0.05
MAX_BASELINE_FALL_RATE = 0.9


def baseline_context(store: RunStore, suite_name: str, metrics: dict, gait: str = "opencat_trF") -> tuple[dict, bool]:
    """Ratio metrics vs the cached firmware gait ON THIS SUITE. Returns (context, group_enabled)."""
    context: dict = {}
    base = store.baseline(gait, suite_name)
    if not base or not base.get("metrics"):
        context["baseline_note"] = (f"No {gait} baseline on {suite_name} yet, so the baseline gates were skipped. "
                                    f"Run `python -m trainer.eval.baselines --suite {suite_name}` (or POST /baselines/compute) to enable them.")
        return context, False
    bm = base["metrics"]
    bd = float(bm.get("forward_distance_p50", 0.0))
    bf = float(bm.get("fall_rate", 0.0))
    context["baseline_metrics"] = bm
    context["baseline_name"] = gait
    if bd < MIN_BASELINE_DISTANCE_M or bf >= MAX_BASELINE_FALL_RATE:
        n_ep = int(bm.get("n_episodes", 0) or 0)
        context["baseline_degenerate"] = True
        context["baseline_note"] = (f"The firmware {gait} cannot do this task (travels {bd:.2f} m, falls "
                                    f"{int(round(bf * n_ep))}/{n_ep}), so there is no baseline to beat on {suite_name}; "
                                    "the ratio gates are not evaluated and the absolute distance gate carries the bar.")
        metrics["baseline_distance_ratio"] = None
        metrics["baseline_fall_delta"] = None
        metrics["baseline_energy_ratio"] = None
        return context, True
    metrics["baseline_distance_ratio"] = float(metrics["forward_distance_p50"] / max(bd, 1e-6))
    metrics["baseline_fall_delta"] = float(metrics["fall_rate"] - bf)
    be = float(bm.get("energy_proxy_w", 0.0))
    if be > 0:
        metrics["baseline_energy_ratio"] = float(metrics["energy_proxy_w"] / be)
    return context, True


def benchmark_run(store: RunStore, run_id: str, *, suite_name: str | None = None, n_episodes: int | None = None,
                  dr_sweep: bool = False, dual_sim: bool = False, record_n: int = 3) -> dict:
    cfg = TrainConfig.model_validate(store.config(run_id)).resolved()
    suite_name = suite_name or store.run_suite(run_id)
    suite = load_suite(suite_name)
    proto = suite_protocol(suite)
    n = int(n_episodes or proto["n_episodes"])
    seed0 = int(proto["seed_start"])
    seconds = float(proto["episode_seconds"])
    cmd = [float(x) for x in proto["command"]]
    common = protocol_kwargs(proto)  # the suite's fixed terrain + spawn jitter + XML variant
    flat_protocol = proto["terrain"].get("kind", "flat") == "flat"
    policy = NumpyPolicy.load(store.policy_path(run_id))
    ctrl = PolicyController(policy)
    groups: set[str] = set()

    # 1. nominal protocol
    stats, rollouts = evaluate_protocol(cfg, ctrl, n_episodes=n, seed_start=seed0, seconds=seconds, command=cmd,
                                        record_seeds=set(range(seed0, seed0 + record_n)),
                                        source={"run_id": run_id, "suite": suite_name, "terrain": proto["terrain"]},
                                        **common)
    metrics = aggregate(stats, seconds)

    # 2. stand still (5 s, zero command) on the same terrain
    ss, _ = evaluate_protocol(cfg, ctrl, n_episodes=max(3, n // 4), seed_start=seed0 + 100, seconds=5.0,
                              command=[0.0, 0.0, 0.0], **common)
    metrics["stand_still_drift_m"] = float(np.mean([abs(s.distance_x) + abs(s.lateral_y) for s in ss]))
    metrics["stand_still_falls"] = int(sum(1 for s in ss if s.fell))

    # 3. multi command (stage >= 1): the same direction at three speeds
    if cfg.curriculum_stage >= 1 and abs(cmd[0]) > 0:
        rmses = []
        sign = 1.0 if cmd[0] > 0 else -1.0
        for v in (0.05, 0.10, 0.18):
            st, _ = evaluate_protocol(cfg, ctrl, n_episodes=max(3, n // 4), seed_start=seed0 + 200, seconds=seconds,
                                      command=[sign * v, 0.0, 0.0], **common)
            rmses.append(aggregate(st, seconds)["vel_tracking_rmse"])
        metrics["multi_command_rmse_max"] = float(max(rmses))
        metrics["multi_command_rmse"] = rmses

    # 4. DR sweep (same terrain)
    if dr_sweep:
        groups.add("dr_sweep")
        sweep = {}
        for name in DR_PRESETS:
            mp, ep = preset_params(name)
            st, _ = evaluate_protocol(cfg, ctrl, n_episodes=max(5, n // 2), seed_start=seed0 + 300, seconds=seconds,
                                      command=cmd, model_params=mp, episode_params=ep, **common)
            agg = aggregate(st, seconds)
            sweep[name] = {"fall_rate": agg["fall_rate"], "distance_p50": agg["forward_distance_p50"]}
        metrics["dr_sweep"] = sweep
        metrics["dr_fall_rate_max"] = float(max(v["fall_rate"] for v in sweep.values()))
        nominal_d = max(metrics["forward_distance_p50"], 1e-6)
        metrics["dr_distance_ratio_min"] = float(min(v["distance_p50"] for v in sweep.values()) / nominal_d)

    # 5. baseline comparison: the task's reference gait replayed on THIS suite's protocol
    gait = (TASKS[cfg.task].baseline if cfg.task in TASKS else None) or "opencat_trF"
    context, enabled = baseline_context(store, suite_name, metrics, gait=gait)
    if enabled:
        groups.add("baseline")

    # 5b. context for the reflection: parent run's gate metrics (SAME suite only) + reward-term
    #     breakdown of the last training eval
    parent = store.state(run_id).get("parent")
    if parent and store.exists(parent):
        prep = store.benchmark(parent, suite_name)
        if prep and prep.get("metrics"):
            context["parent_metrics"] = prep["metrics"]
            context["parent_run_id"] = parent
        else:
            psuite = store.run_suite(parent)
            context["parent_suite_mismatch"] = {"parent_run_id": parent, "parent_suite": psuite, "this_suite": suite_name}
            context["parent_note"] = (f"Parent {parent} has no {suite_name} benchmark (it was benchmarked on {psuite}), "
                                      f"so there is no per-gate parent comparison. Re-benchmark it with "
                                      f"bittle_benchmark_policy(run_id=\"{parent}\", suite=\"{suite_name}\", force=true) "
                                      "if you want the delta.")
    curve = store.curve(run_id, limit=1000)
    if curve:
        last = curve[-1]
        context["reward_breakdown"] = {k.replace("reward/", ""): float(v) for k, v in last.items()
                                       if k.startswith("reward/") and not k.endswith("_std")}

    # 6. dual-sim consistency (GPU engine vs CPU, shared seeds) — flat protocols only: the GPU
    #    replay has no fixed-terrain override yet
    if dual_sim and not flat_protocol:
        metrics["dual_sim_skipped"] = "dual_sim is only defined for flat protocols"
    elif dual_sim:
        try:
            from .dual_sim import gpu_distances

            groups.add("dual_sim")
            gpu_d = gpu_distances(cfg, policy, seeds=list(range(seed0, seed0 + 5)), seconds=seconds, command_vx=cmd[0])
            cpu_d = np.median([s.distance_x for s in stats[:5]])
            metrics["dual_sim_gpu_distance_p50"] = float(np.median(gpu_d))
            metrics["dual_sim_distance_ratio"] = float(np.median(gpu_d) / max(cpu_d, 1e-6))
        except Exception as exc:  # GPU unavailable in this process: report, don't fail the benchmark
            metrics["dual_sim_error"] = str(exc)[:300]

    report = evaluate_gates(metrics, suite, curriculum_stage=cfg.curriculum_stage, groups_enabled=groups, context=context)
    report.update({"run_id": run_id, "n_episodes": n, "seeds": [s.seed for s in stats],
                   "sim": f"mujoco-cpu/bittle_{common['variant']}.xml", "protocol": proto,
                   "task": cfg.task, "episodes": [s.to_dict() for s in stats], "config_hash": cfg.config_hash()})
    store.write_benchmark(run_id, suite["suite"], str(suite["version"]), report)
    for seed, ro in rollouts.items():
        p = store.rollout_path(run_id, suite["suite"], str(suite["version"]), seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ro))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suite", default=None, help="gate suite; default = the run's own suite (its task)")
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
