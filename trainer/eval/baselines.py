"""Baselines: OpenCat gaits and bittle-agent builtin movesets replayed through the evaluator.

A trained policy has to beat ``trF`` (the firmware trot) in the same sim on
the same protocol; the reports are cached under ``runs/baselines/<gait>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..config import TrainConfig
from ..store.runs import RunStore
from .evaluator import GaitController, StandController, aggregate, evaluate_protocol
from .gates import evaluate_gates, load_suite

REPO = Path(__file__).resolve().parent.parent.parent

OPENCAT_GAITS = ("trF", "wkF", "crF")
BUILTIN_GAITS = ("trF", "wkF")


def _builtin_frames(name: str) -> list[dict[str, Any]]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from app.motion.builtin_library import BUILTIN_MOVESETS  # noqa: WPS433 (repo import)

    return BUILTIN_MOVESETS[name]["frames"]


def baseline_controllers(control_hz: int) -> dict[str, Any]:
    out: dict[str, Any] = {"stand": StandController()}
    for g in OPENCAT_GAITS:
        out[f"opencat_{g}"] = GaitController.from_opencat(g)
    for g in BUILTIN_GAITS:
        try:
            out[f"builtin_{g}"] = GaitController.from_keyframes(_builtin_frames(g), control_hz=control_hz)
        except Exception as exc:  # the repo moveset library is optional here
            print(f"builtin {g} unavailable: {exc}")
    return out


def compute_baselines(store: RunStore, *, suite_name: str = "flat_v1", n_episodes: int | None = None,
                      names: list[str] | None = None) -> dict[str, Any]:
    cfg = TrainConfig()
    suite = load_suite(suite_name)
    proto = suite["protocol"]
    n = int(n_episodes or proto["n_episodes"])
    results = {}
    for name, ctrl in baseline_controllers(cfg.control_hz).items():
        if names and name not in names:
            continue
        stats, rollouts = evaluate_protocol(
            cfg, ctrl, n_episodes=n, seed_start=int(proto["seed_start"]), seconds=float(proto["episode_seconds"]),
            command_vx=float(proto["command_vx"]), envelope_tier="tested", record_seeds={int(proto["seed_start"])},
            source={"baseline": name, "suite": suite_name})
        metrics = aggregate(stats, float(proto["episode_seconds"]))
        report = evaluate_gates(metrics, suite, curriculum_stage=0)
        report.update({"baseline": name, "n_episodes": n, "seeds": [s.seed for s in stats],
                       "episodes": [s.to_dict() for s in stats]})
        store.write_baseline(name, report)
        for seed, ro in rollouts.items():
            p = store.baseline_dir(name) / "rollouts" / f"seed_{seed}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            import json
            p.write_text(json.dumps(ro))
        results[name] = {"score": report["score"], "gates_passed": report["gates_passed"],
                         "distance_p50": metrics["forward_distance_p50"], "fall_rate": metrics["fall_rate"]}
        print(f"baseline {name}: dist_p50={metrics['forward_distance_p50']:.3f} m fall={metrics['fall_rate']:.2f} score={report['score']:.2f}")
    return results


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.getenv("TRAINER_RUNS_DIR", str(REPO / "runs")))
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args()
    compute_baselines(RunStore(a.runs_dir), n_episodes=a.n_episodes, names=a.names)
