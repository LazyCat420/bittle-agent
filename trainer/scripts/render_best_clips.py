"""Render ``best.gif`` for benchmarked runs that do not have one yet (or all with ``--force``).

    PYTHONPATH=. .venv-trainer/bin/python trainer/scripts/render_best_clips.py --all [--force]
    PYTHONPATH=. .venv-trainer/bin/python trainer/scripts/render_best_clips.py --run-id <id>

Runs benchmarked before 2026-09-11 have no ``best_seed`` in their report; the best episode is taken
from the report's episode table the same way the benchmark does now, and the report is updated.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trainer.eval.render import best_episode_seed, render_best_clip  # noqa: E402
from trainer.store.runs import RunStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs-dir", default=os.getenv("TRAINER_RUNS_DIR", str(REPO / "runs")))
    a = ap.parse_args(argv)
    store = RunStore(a.runs_dir)
    ids = list(a.run_id)
    if a.all:
        ids = [r["run_id"] for r in store.list_runs() if r.get("status") == "done"]
    n = 0
    for rid in ids:
        suite = store.run_suite(rid)
        rep = store.benchmark(rid, suite)
        if not rep:
            print(f"{rid}: no benchmark"); continue
        out = store.benchmark_dir(rid, suite, str(rep["suite_version"])) / "best.gif"
        if out.is_file() and not a.force:
            print(f"{rid}: has best.gif"); continue
        if rep.get("best_seed") is None:
            rep["best_seed"] = best_episode_seed(rep)
        clip = render_best_clip(store, rid, rep)
        if clip:
            rep["best_clip"] = clip
            store.write_benchmark(rid, suite, str(rep["suite_version"]), rep)
            n += 1
            print(f"{rid}: {clip}")
    print(f"rendered {n} clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
