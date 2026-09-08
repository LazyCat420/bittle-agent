"""Job manager: training and benchmark jobs run as SUBPROCESSES.

Why subprocesses: a JAX/Warp crash must not take the service down, GPU memory
is released when the process exits, and CPU benchmarks can be pinned and
niced independently. The service only watches ``state.json`` and the process
handle. One training job at a time (the GPU is exclusive); benchmarks run in a
small pool. ``TRAINER_FAKE_JOBS=1`` swaps the subprocess for an in-process
fake that writes plausible state quickly (used by the contract tests).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from .runs import RunStore


def _cpu_affinity(cores: str | None) -> list[int] | None:
    """Parse '0-7' or '0,1,2' -> [0..7]; None -> no pinning."""
    if not cores:
        return None
    out: list[int] = []
    for part in cores.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out or None


class JobManager:
    def __init__(self, store: RunStore, *, repo_root: Path, fake: bool = False,
                 cpu_cores: str | None = None, bench_workers: int = 2, python: str | None = None):
        self.store = store
        self.repo_root = Path(repo_root)
        self.fake = fake
        self.cpu_cores = _cpu_affinity(cpu_cores)
        self.python = python or sys.executable
        self._train_q: deque[str] = deque()
        self._bench_q: deque[tuple[str, dict[str, Any]]] = deque()
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._bench_workers = max(1, bench_workers)
        self._bench_active = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="trainer-jobs", daemon=True)
        self._fake_hooks: dict[str, Callable[..., None]] = {}
        self._thread.start()

    # ── public ─────────────────────────────────────────────────────────
    def submit_train(self, run_id: str) -> None:
        with self._lock:
            self._train_q.append(run_id)

    def submit_benchmark(self, run_id: str, opts: dict[str, Any]) -> None:
        with self._lock:
            self._bench_q.append((run_id, opts))
        # a re-benchmark of a 'done' run goes back to 'trained' first, otherwise a waiter would see the
        # OLD 'done' and read a report that does not exist yet (cross-suite force benchmarks hit this)
        fields: dict[str, Any] = {"benchmark_pending": True}
        if self.store.state(run_id).get("status") == "done":
            fields["status"] = "trained"
        self.store.update_state(run_id, **fields)

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._train_q:
                self._train_q.remove(run_id)
                self.store.update_state(run_id, status="cancelled")
                return True
            proc = self._procs.get(run_id)
        if proc and proc.poll() is None:
            proc.terminate()
            self.store.update_state(run_id, status="cancelled", error="cancelled by request")
            return True
        return False

    def active(self) -> dict[str, Any]:
        with self._lock:
            return {
                "train_queue": list(self._train_q),
                "bench_queue": [r for r, _ in self._bench_q],
                "running": [r for r, p in self._procs.items() if p.poll() is None],
            }

    def wait_for(self, run_id: str, predicate: Callable[[dict[str, Any]], bool], wait_s: float) -> dict[str, Any]:
        """Long-poll: return the state as soon as ``predicate`` holds or wait_s elapses."""
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            st = self.store.state(run_id)
            if predicate(st) or time.monotonic() >= deadline:
                return st
            time.sleep(0.25 if self.fake else 1.0)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            procs = list(self._procs.values())
        for p in procs:
            if p.poll() is None:
                p.terminate()

    # ── internals ──────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._reap()
            self._maybe_start_train()
            self._maybe_start_bench()
            time.sleep(0.2 if self.fake else 1.0)

    def _reap(self) -> None:
        with self._lock:
            done = [(r, p) for r, p in self._procs.items() if p.poll() is not None]
            for r, _ in done:
                self._procs.pop(r, None)
        for run_id, proc in done:
            st = self.store.state(run_id)
            if proc.returncode != 0 and st.get("status") not in ("cancelled", "done", "trained"):
                self.store.update_state(run_id, status="failed",
                                        error=f"{st.get('status')} subprocess exited {proc.returncode}; see train.log")
            if getattr(proc, "kind", None) == "bench" and st.get("status") == "benchmarking" and proc.returncode == 0:
                # the benchmark subprocess sets 'done' itself; guard for a silent exit
                if self.store.state(run_id).get("status") == "benchmarking":
                    self.store.update_state(run_id, status="trained", error="benchmark exited without a report")
            with self._lock:
                if st.get("status") == "benchmarking" or "bench" in st.get("job_kind", ""):
                    self._bench_active = max(0, self._bench_active - 1)

    def _train_running(self) -> bool:
        with self._lock:
            return any(self.store.state(r).get("job_kind") == "train" and p.poll() is None
                       for r, p in self._procs.items())

    def _maybe_start_train(self) -> None:
        if self._train_running():
            return
        with self._lock:
            if not self._train_q:
                return
            run_id = self._train_q.popleft()
        self._launch(run_id, "train", ["-m", "trainer.train.run", "--run-id", run_id])

    def _maybe_start_bench(self) -> None:
        with self._lock:
            if self._bench_active >= self._bench_workers or not self._bench_q:
                return
            run_id, opts = self._bench_q.popleft()
            self._bench_active += 1
        suite = str(opts.get("suite") or self.store.run_suite(run_id))
        args = ["-m", "trainer.eval.benchmark_cli", "--run-id", run_id, "--suite", suite,
                "--n-episodes", str(int(opts.get("n_episodes") or 20))]
        if opts.get("dr_sweep"):
            args.append("--dr-sweep")
        if opts.get("dual_sim"):
            args.append("--dual-sim")
        self.store.update_state(run_id, benchmark_pending=False)
        self._launch(run_id, "bench", args, dict(opts, suite=suite))

    def _launch(self, run_id: str, kind: str, args: list[str], opts: dict[str, Any] | None = None) -> None:
        self.store.update_state(run_id, job_kind=kind, status="training" if kind == "train" else "benchmarking")
        if self.fake:
            hook = self._fake_hooks.get(kind)
            t = threading.Thread(target=self._fake_job, args=(run_id, kind, hook, opts or {}), daemon=True)
            t.start()
            proc = _ThreadProc(t)
        else:
            env = dict(os.environ)
            env.setdefault("PYTHONPATH", str(self.repo_root))
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
            if kind == "bench":
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("JAX_PLATFORMS", "cpu")
            log = open(self.store.run_dir(run_id) / ("train.log" if kind == "train" else "bench.log"), "a")
            cmd = [self.python, *args]
            preexec = None
            if self.cpu_cores and hasattr(os, "sched_setaffinity"):
                cores = self.cpu_cores

                def preexec() -> None:  # noqa: E306
                    os.nice(10)
                    try:
                        os.sched_setaffinity(0, cores)
                    except OSError:
                        pass
            proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    preexec_fn=preexec)
            self.store.update_state(run_id, pid=proc.pid)
        proc.kind = kind  # type: ignore[attr-defined]
        with self._lock:
            self._procs[run_id] = proc  # type: ignore[assignment]

    def _fake_job(self, run_id: str, kind: str, hook: Callable[..., None] | None, opts: dict[str, Any] | None = None) -> None:
        """Deterministic stand-in used by the contract tests."""
        try:
            if hook:
                hook(self.store, run_id)
                return
            if kind == "train":
                total = int(self.store.config(run_id).get("ppo", {}).get("num_timesteps", 1000))
                for i in range(1, 4):
                    time.sleep(0.05)
                    self.store.update_state(run_id, progress={"step": int(total * i / 3), "total": total,
                                                              "elapsed_s": 0.05 * i, "eta_s": 0.05 * (3 - i)})
                    self.store.append_curve(run_id, {"step": int(total * i / 3), "reward": 1.0 * i})
                self.store.write_metrics(run_id, {"reward_mean": 3.0, "episode_len": 500, "steps_per_s": 12345.0,
                                                  "vram_gb": 0.0, "fake": True})
                self.store.update_state(run_id, status="trained")
            else:
                from ..eval.gates import evaluate_gates, load_suite
                suite = load_suite(str(opts.get("suite") or self.store.run_suite(run_id)))
                cfg = self.store.config(run_id)
                metrics = {"fall_rate": 0.05, "forward_distance_p50": 0.8, "vel_tracking_rmse": 0.03,
                           "heading_yaw_deg": 4.0, "lateral_drift_m": 0.02, "joint_saturation_pct": 1.0,
                           "action_smoothness_deg": 2.0, "mean_tilt_deg": 3.0, "rms_vz": 0.01,
                           "energy_proxy_w": 0.3, "stand_still_drift_m": 0.01, "stand_still_falls": 0,
                           "peak_joint_speed_rad_s": 3.0, "stall_fraction": 0.001, "stall_concurrent_max": 1,
                           "climb_height_p50": 0.06, "progress_ratio": 0.7, "stumble_rate": 0.02,
                           "foot_clearance_p50_mm": 11.0, "body_clearance_min_mm": 30.0,
                           "n_episodes": 20, "episode_seconds": 10, "fake": True}
                report = evaluate_gates(metrics, suite, curriculum_stage=int(cfg.get("curriculum_stage", 0)))
                report.update({"run_id": run_id, "n_episodes": 20, "seeds": list(range(20)), "sim": "fake"})
                self.store.write_benchmark(run_id, suite["suite"], str(suite["version"]), report)
                rp = self.store.rollout_path(run_id, suite["suite"], str(suite["version"]), 0)
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text('{"schema":"bittle.rollout.v1","fps":50,"dt":0.02,"joint_order":[8,9,10,11,12,13,14,15],"frames":[],"events":[],"summary":{"distance_m":0.8,"fell":false},"source":{"fake":true}}')
                self.store.update_state(run_id, status="done")
        except Exception as exc:  # pragma: no cover - surfaced through state
            self.store.update_state(run_id, status="failed", error=f"fake job error: {exc}")


class _ThreadProc:
    """Minimal Popen look-alike so fake jobs share the reaper path."""

    def __init__(self, thread: threading.Thread):
        self._t = thread
        self.returncode: int | None = None
        self.pid = -1

    def poll(self) -> int | None:
        if self._t.is_alive():
            return None
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        return None
