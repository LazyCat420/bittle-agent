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
from .evaluator import GaitController, StandController
from .gates import evaluate_gates, load_suite
from .scenes import run_suite_scenes

REPO = Path(__file__).resolve().parent.parent.parent

OPENCAT_GAITS = ("trF", "wkF", "crF", "trL", "wkL", "bkF")
#: Replay cadence per gait. 2 rows/step (~100 Hz firmware) is right for the trot, but the open-loop
#: backward walk ALIASES under the sim servo's speed cap: at 2 rows/step it walks FORWARD (+0.39 m),
#: at 1 row/step backward (-0.33 m). Measured 2026-09-07; bkF is replayed at the cadence that goes backward.
ROWS_PER_STEP = {"bkF": 1.0}
BUILTIN_GAITS = ("trF", "wkF")


def _builtin_frames(name: str) -> list[dict[str, Any]]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from app.motion.builtin_library import BUILTIN_MOVESETS  # noqa: WPS433 (repo import)

    return BUILTIN_MOVESETS[name]["frames"]


def baseline_controllers(control_hz: int) -> dict[str, Any]:
    out: dict[str, Any] = {"stand": StandController()}
    for g in OPENCAT_GAITS:
        out[f"opencat_{g}"] = GaitController.from_opencat(g, rows_per_step=ROWS_PER_STEP.get(g))
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
    results = {}
    for name, ctrl in baseline_controllers(cfg.control_hz).items():
        if names and name not in names:
            continue
        # every scene of the suite: the gait is replayed on the SAME ground the policies are judged on
        metrics, stats, rollouts, proto = run_suite_scenes(cfg, ctrl, suite, n_episodes=n_episodes, envelope_tier="tested",
                                                           record_n=1, source={"baseline": name, "suite": suite_name})
        n = len(stats)
        report = evaluate_gates(metrics, suite, curriculum_stage=0)
        report.update({"baseline": name, "suite": suite_name, "n_episodes": n, "seeds": [s.seed for s in stats],
                       "protocol": proto, "episodes": [s.to_dict() for s in stats]})
        store.write_baseline(name, report, suite_name)
        for seed, ro in rollouts.items():
            p = store.baseline_dir(name, suite_name) / "rollouts" / f"seed_{seed}.json"
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
    ap.add_argument("--suite", default="flat_v1", help="replay the gaits on this suite's protocol (terrain, command)")
    a = ap.parse_args()
    compute_baselines(RunStore(a.runs_dir), suite_name=a.suite, n_episodes=a.n_episodes, names=a.names)
