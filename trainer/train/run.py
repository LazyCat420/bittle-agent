"""Training subprocess: ``python -m trainer.train.run --run-id <id>``.

Launched by the JobManager. Reads the run's config, trains, exports the
policy, and flips the state to ``trained`` (or ``failed`` with the error).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from ..config import TrainConfig
from ..store.runs import RunStore

REPO = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default=os.getenv("TRAINER_RUNS_DIR", str(REPO / "runs")))
    ap.add_argument("--impl", default=None, help="override sim impl (warp|jax)")
    ap.add_argument("--sim-dt", type=float, default=float(os.getenv("TRAINER_SIM_DT", "0.002")))
    a = ap.parse_args(argv)
    store = RunStore(a.runs_dir)
    run_dir = store.run_dir(a.run_id)
    cfg = TrainConfig.model_validate(store.config(a.run_id))
    total = cfg.ppo.num_timesteps

    def progress(point: dict) -> None:
        store.append_curve(a.run_id, point)
        eta = None
        if point["step"] > 0 and point.get("steps_per_s"):
            eta = round((total - point["step"]) / max(point["steps_per_s"], 1e-6), 1)
        store.update_state(a.run_id, progress={"step": point["step"], "total": total,
                                               "elapsed_s": point["elapsed_s"], "eta_s": eta,
                                               "reward": point["reward"], "steps_per_s": point.get("steps_per_s")})

    try:
        from .ppo import load_restore_params, train_policy

        restore, why = None, None
        parent = store.state(a.run_id).get("parent")
        if parent and cfg.init_from_parent and store.exists(parent):
            restore, why = load_restore_params(store.run_dir(parent), cfg)
        store.update_state(a.run_id, status="training",
                           warm_start={"parent": parent, "used": restore is not None, "reason": why})
        metrics = train_policy(cfg, run_dir, progress=progress, impl=a.impl or cfg.sim_impl, sim_dt=a.sim_dt,
                               restore_params=restore)
        store.write_metrics(a.run_id, {k: v for k, v in metrics.items() if k != "curve"})
        store.update_state(a.run_id, status="trained")
        return 0
    except Exception as exc:
        traceback.print_exc()
        store.update_state(a.run_id, status="failed", error=f"training failed: {exc}"[:800])
        return 1


if __name__ == "__main__":
    # Serialize CUDA ownership with the local music renderer. The service launches
    # a fresh process for each run, so this also covers an already-running service.
    import fcntl
    with open("/tmp/sun-rtx3090ti.gpu.lock", "a") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        sys.exit(main())
