"""Run store: one directory per run, a rebuilt index, and a leaderboard.

Layout::

    runs/
      index.json                      rebuilt from the directories on demand
      baselines/<gait>/report.json    replayed OpenCat gaits (computed once)
      <run_id>/
        config.json                   resolved TrainConfig
        state.json                    {status, progress, error, timestamps}
        metrics.json                  training metrics (reward curve summary)
        curves.jsonl                  one line per eval point
        policy/policy.npz             exported weights + normaliser + layout
        benchmark/<suite>@<ver>/report.json
        benchmark/<suite>@<ver>/rollouts/seed_N.json
        train.log

The store is the source of truth; bittle-agent only caches ids.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("queued", "training", "trained", "benchmarking", "done", "failed", "cancelled")


def _now() -> str:
    """MILLISECOND resolution on purpose: the dashboard orders runs by when they finished, and two runs
    that finish inside the same second are otherwise tied with nothing meaningful to break the tie (a
    run id's suffix is random, not a sequence). ISO-8601 still sorts lexicographically, and a stamp
    written at the old second resolution sorts first within its second, which is harmless."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


class RunStore:
    def __init__(self, root: str | Path):
        #: update_state is a read-modify-write on state.json; the job loop, the fake job
        #: threads and request handlers all call it, so serialise it in-process.
        self._lock = threading.RLock()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "baselines").mkdir(exist_ok=True)

    # ── paths ──────────────────────────────────────────────────────────
    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"bad run_id {run_id!r}")
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.run_dir(run_id) / "state.json").is_file()

    def benchmark_dir(self, run_id: str, suite: str, version: str) -> Path:
        return self.run_dir(run_id) / "benchmark" / f"{suite}@{version}"

    def policy_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "policy" / "policy.npz"

    # ── lifecycle ──────────────────────────────────────────────────────
    def create(self, config: dict[str, Any], *, name: str = "", parent: str | None = None, notes: str = "",
               config_hash: str = "", task: str | None = None, suite: str | None = None) -> str:
        """``task``/``suite`` default from the config (the task decides the suite); the three runs
        shipped before suites existed have neither key and read as flat_walk / flat_v1."""
        from ..tasks import default_suite_for, task_name_for

        run_id = new_run_id()
        d = self.run_dir(run_id)
        d.mkdir(parents=True)
        _write_json(d / "config.json", config)
        task = task or task_name_for(config)
        _write_json(d / "state.json", {
            "run_id": run_id,
            "name": name or run_id,
            "parent": parent,
            "notes": notes,
            "task": task,
            "suite": suite or default_suite_for(task),
            "config_hash": config_hash,
            "status": "queued",
            "progress": {"step": 0, "total": int(config.get("ppo", {}).get("num_timesteps", 0)), "elapsed_s": 0.0, "eta_s": None},
            "error": None,
            "created": _now(),
            "updated": _now(),
            "started": None,
            "finished": None,
            "pid": None,
        })
        return run_id

    def state(self, run_id: str) -> dict[str, Any]:
        st = _read_json(self.run_dir(run_id) / "state.json")
        if st is None:
            raise KeyError(run_id)
        return st

    def update_state(self, run_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            st = self.state(run_id)
            st.update(fields)
            st["updated"] = _now()
            if fields.get("status") == "training" and not st.get("started"):
                st["started"] = _now()
            if fields.get("status") in ("done", "failed", "cancelled", "trained"):
                st["finished"] = _now()
            _write_json(self.run_dir(run_id) / "state.json", st)
            return st

    def config(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "config.json", {})

    def run_task(self, run_id: str) -> str:
        return str(self.state(run_id).get("task") or "flat_walk")

    def run_suite(self, run_id: str) -> str:
        """The suite this run is judged on: recorded at creation, flat_v1 for pre-suite runs."""
        return str(self.state(run_id).get("suite") or "flat_v1")

    def metrics(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "metrics.json", {})

    def write_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        _write_json(self.run_dir(run_id) / "metrics.json", metrics)

    def append_curve(self, run_id: str, point: dict[str, Any]) -> None:
        with open(self.run_dir(run_id) / "curves.jsonl", "a") as fh:
            fh.write(json.dumps(point) + "\n")

    def curve(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        path = self.run_dir(run_id) / "curves.jsonl"
        if not path.is_file():
            return []
        pts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(pts) <= limit:
            return pts
        # keep first, last and an even subsample in between
        idx = sorted({0, len(pts) - 1} | {int(i * (len(pts) - 1) / (limit - 1)) for i in range(limit)})
        return [pts[i] for i in idx]

    # ── benchmarks ─────────────────────────────────────────────────────
    def write_benchmark(self, run_id: str, suite: str, version: str, report: dict[str, Any]) -> Path:
        d = self.benchmark_dir(run_id, suite, version)
        _write_json(d / "report.json", report)
        return d / "report.json"

    def benchmark(self, run_id: str, suite: str, version: str | None = None) -> dict[str, Any] | None:
        base = self.run_dir(run_id) / "benchmark"
        if not base.is_dir():
            return None
        if version:
            return _read_json(base / f"{suite}@{version}" / "report.json")
        candidates = sorted(base.glob(f"{suite}@*/report.json"))
        return _read_json(candidates[-1]) if candidates else None

    def rollout_path(self, run_id: str, suite: str, version: str, seed: int) -> Path:
        return self.benchmark_dir(run_id, suite, version) / "rollouts" / f"seed_{seed}.json"

    # ── baselines (per suite; the flat_v1 layout predates suites and stays where it was) ──
    def baseline_dir(self, gait: str, suite: str | None = "flat_v1") -> Path:
        if not suite or suite == "flat_v1":
            return self.root / "baselines" / gait
        return self.root / "baselines" / gait / suite

    def write_baseline(self, gait: str, report: dict[str, Any], suite: str | None = "flat_v1") -> None:
        _write_json(self.baseline_dir(gait, suite) / "report.json", report)

    def baseline(self, gait: str, suite: str | None = "flat_v1") -> dict[str, Any] | None:
        return _read_json(self.baseline_dir(gait, suite) / "report.json")

    def baselines(self, suite: str | None = "flat_v1") -> dict[str, Any]:
        out = {}
        base = self.root / "baselines"
        for d in sorted(base.iterdir()):
            if d.is_dir():
                rep = self.baseline(d.name, suite)
                if rep:
                    out[d.name] = rep
        return out

    # ── index / leaderboard ────────────────────────────────────────────
    def summary(self, run_id: str) -> dict[str, Any]:
        """One leaderboard row, scored on the run's OWN suite."""
        st = self.state(run_id)
        suite = self.run_suite(run_id)
        bench = self.benchmark(run_id, suite)
        m = bench.get("metrics", {}) if bench else {}
        return {
            "run_id": run_id,
            "name": st.get("name"),
            "status": st.get("status"),
            "created": st.get("created"),
            # when the run actually FINISHED (its benchmark landed), and the operator's own note saying
            # what this run was trying to find out -- both are what the dashboard orders and explains by
            "finished": st.get("finished"),
            "notes": st.get("notes") or "",
            "parent": st.get("parent"),
            "task": self.run_task(run_id),
            "suite": suite,
            "config_hash": st.get("config_hash"),
            "score": bench.get("score") if bench else None,
            "gates_passed": bench.get("gates_passed") if bench else None,
            "gates_total": bench.get("gates_total") if bench else None,
            "suite_version": bench.get("suite_version") if bench else None,
            "dist_p50": m.get("forward_distance_p50"),
            "fall_rate": m.get("fall_rate"),
            "error": st.get("error"),
        }

    #: sort=finished puts the most recently COMPLETED run first and keeps anything still running or
    #: queued at the very top (they have no finish time yet but are the runs an operator is watching).
    SORTS = ("created", "score", "finished")

    def list_runs(self, *, sort: str = "created", limit: int = 20, suite: str | None = None) -> list[dict[str, Any]]:
        """``suite`` filters to runs judged on that suite. Scores from different suites are NOT
        comparable, so a score sort without a suite groups by (suite, version) first.

        ``sort``: ``created`` (newest submitted), ``score`` (the leaderboard), ``finished`` (newest
        completed first, with in-flight runs pinned above them)."""
        rows = []
        for d in self.root.iterdir():
            if d.is_dir() and (d / "state.json").is_file():
                try:
                    rows.append(self.summary(d.name))
                except Exception:  # corrupt dir: skip, never crash the listing
                    continue
        if suite:
            rows = [r for r in rows if r["suite"] == suite]
        if sort == "score":
            rows.sort(key=lambda r: (r["suite"] or "", r["suite_version"] or "", (r["gates_passed"] or 0),
                                     (r["score"] or -1e9), (r["dist_p50"] or 0)), reverse=True)
        elif sort == "finished":
            busy = {"queued", "training", "benchmarking", "trained"}
            # the timestamps have SECOND resolution, so two runs finishing in the same second tie; the
            # run id starts with its creation timestamp, which breaks the tie as "newer run first"
            # rather than leaving the order to however the directory happened to be read
            rows.sort(key=lambda r: (r["status"] in busy, r["finished"] or r["created"] or "", r["run_id"]),
                      reverse=True)
        else:
            rows.sort(key=lambda r: r["created"] or "", reverse=True)
        if not suite:
            _write_json(self.root / "index.json", {"updated": _now(), "runs": rows})
        return rows[:limit]

    def best(self, suite: str = "flat_v1") -> dict[str, Any] | None:
        """Best run on ``suite``, ranked only within the HIGHEST suite version present; older-version
        runs are reported under ``stale_versions`` rather than mixed into the ranking."""
        rows = [r for r in self.list_runs(sort="score", limit=1000, suite=suite) if r["score"] is not None]
        if not rows:
            return None
        newest = max(r["suite_version"] or "" for r in rows)
        current = [r for r in rows if (r["suite_version"] or "") == newest]
        stale = sorted({r["suite_version"] for r in rows if (r["suite_version"] or "") != newest})
        top = dict(current[0])
        if stale:
            top["stale_versions"] = stale
        return top

    def best_by_suite(self) -> dict[str, dict[str, Any]]:
        suites = sorted({r["suite"] for r in self.list_runs(limit=1000)})
        return {s: b for s in suites if (b := self.best(s)) is not None}

    def all_gates_pass_run(self, suite: str) -> str | None:
        """The best run on ``suite`` that passes every evaluated gate (a task prerequisite), or None."""
        for r in self.list_runs(sort="score", limit=1000, suite=suite):
            if r["gates_total"] and r["gates_passed"] == r["gates_total"]:
                return r["run_id"]
        return None

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)

    def mark_orphans_failed(self) -> list[str]:
        """On service restart, anything still 'training'/'benchmarking' is dead."""
        orphaned = []
        for d in self.root.iterdir():
            if not (d.is_dir() and (d / "state.json").is_file()):
                continue
            st = self.state(d.name)
            if st.get("status") in ("queued", "training", "benchmarking"):
                self.update_state(d.name, status="failed", error="orphaned: trainer restarted while the job was running")
                orphaned.append(d.name)
        return orphaned
