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
from .scenes import run_suite_scenes

REPO = Path(__file__).resolve().parent.parent.parent

#: A firmware-gait baseline that travels less than this (or falls in >= 90% of episodes) on a
#: protocol is DEGENERATE: its distance ratio would hand out a free pass, so the ratio gates are
#: reported as not evaluated instead. The absolute distance gate then carries the load.
MIN_BASELINE_DISTANCE_M = 0.05
MAX_BASELINE_FALL_RATE = 0.9



def terrain_gap(cfg, proto: dict[str, Any]) -> dict[str, Any]:
    """How the run's TRAINING terrain compares to the suite protocol it is judged on."""
    pt = proto.get("terrain") or {"kind": "flat"}
    t = cfg.terrain
    gap: dict[str, Any] = {"train_kind": t.kind, "protocol_kind": pt.get("kind", "flat")}
    if pt.get("kind", "flat") in ("rough", "rough_slope"):
        h = float(pt.get("box_height_m", 0.0))
        gap.update({"protocol_box_height_m": h, "train_box_height_m": [float(t.box_height_m[0]), float(t.box_height_m[1])],
                    "protocol_n_boxes": int(pt.get("n_boxes", 0)), "train_n_boxes": int(t.n_boxes),
                    "protocol_box_spacing_m": float(pt.get("box_spacing_m", 0.10)), "train_box_spacing_m": float(t.box_spacing_m),
                    # the share of training boxes at least as tall as the protocol's (uniform sampling)
                    "train_share_at_or_above_protocol_height": (0.0 if t.n_boxes == 0 or t.box_height_m[1] <= h else
                                                                round(float((t.box_height_m[1] - max(h, t.box_height_m[0])) / max(t.box_height_m[1] - t.box_height_m[0], 1e-9)), 3))})
        gap["easier_than_protocol"] = bool(t.n_boxes < int(pt.get("n_boxes", 0)) or t.box_height_m[1] < h
                                           or gap["train_share_at_or_above_protocol_height"] < 0.25)
    if pt.get("kind", "flat") == "stairs":
        rise, steps = float(pt.get("stair_rise_m", 0.0)), int(pt.get("stair_steps", 0))
        gap.update({"protocol_stair_rise_m": rise, "protocol_stair_steps": steps,
                    "train_stair_rise_m": [float(t.stair_rise_m[0]), float(t.stair_rise_m[1])],
                    "train_stair_steps": [int(t.stair_steps[0]), int(t.stair_steps[1])],
                    "train_share_at_or_above_protocol_rise": (0.0 if t.kind != "stairs" or t.stair_rise_m[1] <= rise else
                                                              round(float((t.stair_rise_m[1] - max(rise, t.stair_rise_m[0]))
                                                                          / max(t.stair_rise_m[1] - t.stair_rise_m[0], 1e-9)), 3))})
        gap["easier_than_protocol"] = bool(t.kind != "stairs" or t.stair_rise_m[1] < rise or t.stair_steps[1] < steps
                                           or gap["train_share_at_or_above_protocol_rise"] < 0.25)
    if pt.get("kind", "flat") in ("slope", "rough_slope"):
        sd = float(pt.get("slope_deg", 0.0))
        gap.update({"protocol_slope_deg": sd, "train_slope_deg": [float(t.slope_deg[0]), float(t.slope_deg[1])]})
        gap["easier_than_protocol"] = bool(gap.get("easier_than_protocol", False) or t.slope_deg[1] < sd)
    return gap

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
                  dr_sweep: bool = False, dual_sim: bool = False, record_n: int | None = None,
                  render_clip: bool = True) -> dict:
    """``record_n`` None = record EVERY episode's rollout (so the viewer can replay the best attempt);
    ``render_clip`` writes ``best.gif`` for the best episode next to the report."""
    cfg = TrainConfig.model_validate(store.config(run_id)).resolved()
    suite_name = suite_name or store.run_suite(run_id)
    suite = load_suite(suite_name)
    policy = NumpyPolicy.load(store.policy_path(run_id))
    if cfg.reflex.arm == "conventional":
        from ..policy.conventional_reflex import ConventionalReflex
        from ..policy.wrapper import ReflexAugmentedPolicy
        policy = ReflexAugmentedPolicy(policy, ConventionalReflex(kp_pitch=cfg.reflex.kp_pitch, kd_pitch=cfg.reflex.kd_pitch,
                                                                 kp_roll=cfg.reflex.kp_roll, kd_roll=cfg.reflex.kd_roll,
                                                                 max_trim_deg=cfg.reflex.max_trim_deg),
                                       action_scale_deg=cfg.action_scale_deg)
    elif cfg.reflex.arm == "fly":
        from ..policy.fly_reflex import FlyReflex
        from ..policy.wrapper import ReflexAugmentedPolicy
        policy = ReflexAugmentedPolicy(policy, FlyReflex(haltere_jerk_thresh=cfg.reflex.haltere_jerk_thresh,
                                                        max_trim_deg=cfg.reflex.max_trim_deg),
                                       action_scale_deg=cfg.action_scale_deg)
    elif cfg.reflex.arm == "null":
        from ..policy.null_reflex import NullReflex
        from ..policy.wrapper import ReflexAugmentedPolicy
        policy = ReflexAugmentedPolicy(policy, NullReflex(max_trim_deg=cfg.reflex.max_trim_deg),
                                       action_scale_deg=cfg.action_scale_deg)
    ctrl = PolicyController(policy)
    groups: set[str] = set()

    # 1 + 2. every scene of the suite (one for a classic suite) + its stand-still sub-protocol
    if record_n is None:
        record_n = int(n_episodes or suite_protocol(suite)["n_episodes"])
    metrics, stats, rollouts, proto = run_suite_scenes(cfg, ctrl, suite, n_episodes=n_episodes, record_n=record_n,
                                                       source={"run_id": run_id, "suite": suite_name}, stand_still=True)
    n = int(proto["n_episodes"])
    seed0 = int(proto["seed_start"])
    seconds = float(proto["episode_seconds"])
    cmd = [float(x) for x in proto["command"]]
    common = protocol_kwargs(proto)  # the PRIMARY scene's terrain: what the sub-protocols below run on
    flat_protocol = proto["terrain"].get("kind", "flat") == "flat"

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
        # the training-side distances: brax's eval (random commands, the run's own terrain) and the bar
        # eval (the suite protocol) -- what tells "easier classroom than exam" from "the policy is stuck"
        for key in ("distance_x", "bar_distance_x", "bar_distance_p50", "bar_fall_rate"):
            if last.get(key) is not None and np.isfinite(float(last[key])):
                context[f"train_{key}"] = float(last[key])
    context["reward_weights"] = {k: float(v) for k, v in cfg.reward.weights.model_dump().items()}
    context["terrain_gap"] = terrain_gap(cfg, proto)

    # 6. dual-sim consistency (GPU engine vs CPU, shared seeds) on the protocol's fixed field (the GPU
    #    replay pins the same TerrainField the CPU episodes ran on; scenes suites use the primary scene)
    if dual_sim:
        try:
            from .dual_sim import gpu_distances

            groups.add("dual_sim")
            gpu_d = gpu_distances(cfg, policy, seeds=list(range(seed0, seed0 + 5)), seconds=seconds, command_vx=cmd[0],
                                  terrain=None if flat_protocol else common["terrain"],
                                  spawn_jitter_m=None if flat_protocol else common.get("spawn_jitter_m"))
            cpu_d = np.median([s.distance_x for s in stats if s.scene == proto.get("scene", "")][:5])
            metrics["dual_sim_gpu_distance_p50"] = float(np.median(gpu_d))
            metrics["dual_sim_distance_ratio"] = float(np.median(gpu_d) / max(cpu_d, 1e-6))
        except Exception as exc:  # GPU unavailable in this process: report, don't fail the benchmark
            metrics["dual_sim_error"] = str(exc)[:300]

    report = evaluate_gates(metrics, suite, curriculum_stage=cfg.curriculum_stage, groups_enabled=groups, context=context)
    report.update({"run_id": run_id, "n_episodes": n, "seeds": [s.seed for s in stats],
                   "sim": f"mujoco-cpu/bittle_{common['variant']}.xml", "protocol": proto,
                   "task": cfg.task, "episodes": [s.to_dict() for s in stats], "config_hash": cfg.config_hash()})
    from .render import best_episode_seed

    report["best_seed"] = best_episode_seed(report)
    store.write_benchmark(run_id, suite["suite"], str(suite["version"]), report)
    for seed, ro in rollouts.items():
        p = store.rollout_path(run_id, suite["suite"], str(suite["version"]), seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ro))
    if render_clip:
        try:
            from .render import render_best_clip

            clip = render_best_clip(store, run_id, report)
            if clip:
                report["best_clip"] = clip
                store.write_benchmark(run_id, suite["suite"], str(suite["version"]), report)
        except Exception as exc:  # a clip is a nicety; the benchmark result must land regardless
            print(f"best clip not rendered: {exc!r}")
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
