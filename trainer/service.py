"""bittle-trainer HTTP service (port 8009).

    python -m trainer.service            # venv on the GPU box
    TRAINER_FAKE_JOBS=1 ...              # contract tests / demo without a GPU

Every long operation is an async job with an id; ``?wait_s=`` long-polls so
one LLM tool call can span minutes. The LLM only ever sends config patches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from . import __version__
from .config import TrainConfig, apply_patch, config_diff, default_config, json_schema, validation_errors
from .eval.gates import list_suites, load_suite
from .store.jobs import JobManager
from .store.runs import RunStore
from .tasks import TASKS, catalogue, default_suite_for

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.getenv("TRAINER_RUNS_DIR", str(REPO / "runs")))
FAKE = os.getenv("TRAINER_FAKE_JOBS", "") in ("1", "true", "yes")
MAX_WAIT_S = 300.0

store = RunStore(RUNS_DIR)
jobs = JobManager(store, repo_root=REPO, fake=FAKE, cpu_cores=os.getenv("TRAINER_CPU_CORES", "0-7"),
                  bench_workers=int(os.getenv("TRAINER_BENCH_WORKERS", "2")))
orphans = store.mark_orphans_failed()

app = FastAPI(title="bittle-trainer", version=__version__,
              description="RL locomotion training service for the Petoi Bittle X (MuJoCo Warp + Brax PPO).")


# ── models ─────────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    config_patch: dict[str, Any] = Field(default_factory=dict)
    base_run_id: str | None = None


class SubmitRequest(ValidateRequest):
    name: str = Field("", max_length=80)
    notes: str = Field("", max_length=2000)
    #: task name from GET /tasks; merged into the patch (its suite is recorded on the run)
    task: str | None = None


class BenchmarkRequest(BaseModel):
    #: None = the run's own suite (its task's). A different suite needs ``force`` — cross-suite
    #: numbers are an ablation, never a leaderboard entry.
    suite: str | None = None
    force: bool = False
    n_episodes: int | None = Field(None, ge=1, le=200)
    dr_sweep: bool = False
    dual_sim: bool = False
    wait_s: float = Field(0.0, ge=0.0, le=MAX_WAIT_S)


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(..., min_length=1, max_length=10)


class BaselinesRequest(BaseModel):
    n_episodes: int | None = Field(None, ge=1, le=200)
    names: list[str] | None = None
    suite: str = "flat_v1"


# ── helpers ────────────────────────────────────────────────────────────────

def _resolve(req: ValidateRequest) -> tuple[TrainConfig, dict[str, Any]]:
    base: dict[str, Any] | None = None
    if req.base_run_id:
        if not store.exists(req.base_run_id):
            raise HTTPException(404, f"base_run_id {req.base_run_id!r} not found")
        base = store.config(req.base_run_id)
    patch = dict(req.config_patch or {})
    task = getattr(req, "task", None)
    if task:
        if task not in TASKS:
            raise HTTPException(422, {"errors": [f"unknown task {task!r}; known: {sorted(TASKS)}"]})
        if patch.get("task") not in (None, task):
            raise HTTPException(422, {"errors": [f"task_conflict: task={task!r} but config_patch.task={patch['task']!r}"]})
        # The task's own patch (terrain level, reward defaults, budget) goes UNDER the caller's patch --
        # but ONLY when the base run is not already on this task. A same-task child inherits its parent's
        # tuned values; re-applying the stock defaults silently reset feet_air_time 10 -> 0.3 on r14
        # (2026-09-10, found by GLM), which is exactly the "invisible reset" a warm start must not have.
        same_task = bool(base) and base.get("task") == task
        task_patch = {"task": task} if same_task else TASKS[task].config_patch
        patch = _deep_merge(task_patch, patch)
    try:
        cfg = apply_patch(base, patch)
    except ValidationError as exc:
        raise HTTPException(422, {"errors": validation_errors(exc)})
    if cfg.task not in TASKS:
        raise HTTPException(422, {"errors": [f"unknown task {cfg.task!r}; known: {sorted(TASKS)}"]})
    base_cfg = TrainConfig.model_validate(base) if base else default_config()
    diff = config_diff(base_cfg.resolved().model_dump(mode="json"), cfg.resolved().model_dump(mode="json"))
    return cfg, {"base": req.base_run_id or "defaults", "diff": diff}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if out:
            name, total, used = [x.strip() for x in out[0].split(",")]
            return {"name": name, "memory_total": total, "memory_used": used}
    except Exception:
        pass
    return {"name": None}


def _state_view(run_id: str) -> dict[str, Any]:
    st = store.state(run_id)
    view = dict(st)
    view["config"] = store.config(run_id)
    view["metrics"] = store.metrics(run_id)
    view["curve"] = store.curve(run_id, limit=20)
    view["task"] = store.run_task(run_id)
    view["suite"] = store.run_suite(run_id)
    bench = store.benchmark(run_id, view["suite"])
    if bench:
        view["benchmark"] = {k: v for k, v in bench.items() if k != "episodes"}
    return view


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    import mujoco

    info = {"ok": True, "version": __version__, "fake_jobs": FAKE, "runs_dir": str(RUNS_DIR),
            "mujoco": mujoco.__version__, "gpu": _gpu_info(), "jobs": jobs.active(),
            "orphans_failed_on_boot": orphans, "python": sys.version.split()[0],
            "suites": list_suites(), "tasks_hint": "call GET /tasks for the task catalogue"}
    try:
        import mujoco_warp  # noqa: F401
        info["physics_impl"] = "warp"
    except Exception:
        info["physics_impl"] = "jax"
    return info


@app.get("/config/schema")
def config_schema() -> dict[str, Any]:
    return {"schema_version": default_config().schema_version, "defaults": default_config().model_dump(mode="json"),
            "json_schema": json_schema()}


@app.post("/config/validate")
def config_validate(req: ValidateRequest) -> dict[str, Any]:
    cfg, extra = _resolve(req)
    return {"ok": True, "config": cfg.model_dump(mode="json"), "resolved": cfg.resolved().model_dump(mode="json"),
            "config_hash": cfg.config_hash(), **extra, "errors": [], "warnings": _warnings(cfg)}


def _warnings(cfg: TrainConfig) -> list[str]:
    w = []
    r = cfg.resolved()
    if r.ppo.num_timesteps > 100_000_000:
        w.append("num_timesteps > 100M: expect > 30 min on the 3090 Ti")
    if r.ppo.num_envs < 256:
        w.append("num_envs < 256 underuses the GPU; throughput will be poor")
    if r.reward.weights.tracking_lin_vel == 0:
        w.append("tracking_lin_vel is 0: nothing rewards walking")
    return w


@app.post("/runs", status_code=202)
def submit_run(req: SubmitRequest) -> dict[str, Any]:
    cfg, extra = _resolve(req)
    suite = default_suite_for(cfg.task)
    run_id = store.create(cfg.model_dump(mode="json"), name=req.name, parent=req.base_run_id, notes=req.notes,
                          config_hash=cfg.config_hash(), task=cfg.task, suite=suite)
    jobs.submit_train(run_id)
    return {"run_id": run_id, "status": "queued", "task": cfg.task, "suite": suite,
            "config_hash": cfg.config_hash(), "diff": extra["diff"]}


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    """The task catalogue plus what the store knows: is each prerequisite met, and by which run."""
    out = []
    for t in catalogue():
        suite = t["suite"]
        try:
            t["suite_version"] = str(load_suite(suite).get("version"))
        except ValueError:
            t["suite_version"] = None
        unmet = [p for p in t["prerequisites"] if store.all_gates_pass_run(default_suite_for(p)) is None]
        satisfied_by = {p: store.all_gates_pass_run(default_suite_for(p)) for p in t["prerequisites"]}
        warm = None
        if t["prerequisites"] and not unmet:
            warm = satisfied_by[t["prerequisites"][-1]]
        best_here = store.best(suite)
        if best_here:
            warm = best_here["run_id"]
        t.update({"prereq_satisfied_by": satisfied_by, "warm_start_from": warm,
                  "best_run": best_here["run_id"] if best_here else None,
                  "status": "ready" if not unmet else f"blocked: no {', '.join(unmet)} run passes every gate yet"})
        out.append(t)
    return {"tasks": out, "suites": list_suites()}


@app.get("/runs")
def list_runs(sort: Literal["created", "score"] = "created", limit: int = Query(20, ge=1, le=200),
              suite: str | None = None) -> dict[str, Any]:
    rows = store.list_runs(sort=sort, limit=limit, suite=suite)
    bsuite = suite or "flat_v1"
    baselines = {k: {"score": v.get("score"), "gates_passed": v.get("gates_passed"),
                     "distance_p50": v.get("metrics", {}).get("forward_distance_p50"),
                     "fall_rate": v.get("metrics", {}).get("fall_rate")} for k, v in store.baselines(bsuite).items()}
    return {"runs": rows, "baselines": baselines, "baselines_suite": bsuite, "best": store.best(bsuite),
            "best_by_suite": store.best_by_suite()}


@app.get("/runs/{run_id}")
def get_run(run_id: str, wait_s: float = Query(0.0, ge=0.0, le=MAX_WAIT_S),
            until: Literal["any", "trained", "done"] = "any") -> dict[str, Any]:
    if not store.exists(run_id):
        raise HTTPException(404, f"run {run_id!r} not found")
    if wait_s > 0:
        terminal = {"failed", "cancelled"}
        if until == "trained":
            target = {"trained", "done", "benchmarking"} | terminal
        elif until == "done":
            target = {"done"} | terminal
        else:
            target = {"trained", "done"} | terminal
        jobs.wait_for(run_id, lambda st: st.get("status") in target, wait_s)
    return _state_view(run_id)


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    if not store.exists(run_id):
        raise HTTPException(404, f"run {run_id!r} not found")
    return {"ok": jobs.cancel(run_id), "state": store.state(run_id)}


@app.post("/runs/{run_id}/benchmark", status_code=202)
def benchmark_run(run_id: str, req: BenchmarkRequest) -> Any:
    if not store.exists(run_id):
        raise HTTPException(404, f"run {run_id!r} not found")
    st = store.state(run_id)
    if st["status"] not in ("trained", "done"):
        raise HTTPException(409, f"run {run_id!r} is {st['status']}; benchmark needs a trained policy")
    own = store.run_suite(run_id)
    suite = req.suite or own
    try:
        load_suite(suite)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if suite != own and not req.force:
        raise HTTPException(422, {"error": "suite_task_mismatch", "run_suite": own, "requested": suite,
                                  "hint": "pass force=true to benchmark across suites (an ablation, not a leaderboard entry)"})
    jobs.submit_benchmark(run_id, dict(req.model_dump(), suite=suite))
    if req.wait_s > 0:
        jobs.wait_for(run_id, lambda s: s.get("status") in ("done", "failed", "cancelled")
                      or (s.get("status") == "trained" and s.get("error")), req.wait_s)
    rep = store.benchmark(run_id, suite)
    if rep and store.state(run_id)["status"] == "done":
        return JSONResponse(status_code=200, content=_report_view(rep))
    return {"run_id": run_id, "status": store.state(run_id)["status"], "queued": True}


def _report_view(rep: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rep.items() if k != "episodes"}


@app.get("/runs/{run_id}/benchmark")
def get_benchmark(run_id: str, suite: str | None = None, full: bool = False) -> dict[str, Any]:
    if not store.exists(run_id):
        raise HTTPException(404, f"run {run_id!r} not found")
    suite = suite or store.run_suite(run_id)
    rep = store.benchmark(run_id, suite)
    if not rep:
        raise HTTPException(404, f"no {suite} benchmark for {run_id!r} yet")
    return rep if full else _report_view(rep)


@app.get("/runs/{run_id}/rollout")
def get_rollout(run_id: str, seed: int = 0, suite: str | None = None) -> Any:
    if not store.exists(run_id):
        raise HTTPException(404, f"run {run_id!r} not found")
    suite = suite or store.run_suite(run_id)
    rep = store.benchmark(run_id, suite)
    if not rep:
        raise HTTPException(404, "benchmark the run first")
    path = store.rollout_path(run_id, suite, rep["suite_version"], seed)
    if not path.is_file():
        raise HTTPException(404, f"no recorded rollout for seed {seed}")
    return FileResponse(path, media_type="application/json")


@app.get("/baselines")
def get_baselines(suite: str = "flat_v1") -> dict[str, Any]:
    return {k: _report_view(v) for k, v in store.baselines(suite).items()}


@app.get("/baselines/{name}/rollout")
def get_baseline_rollout(name: str, seed: int = 0, suite: str = "flat_v1") -> Any:
    path = store.baseline_dir(name, suite) / "rollouts" / f"seed_{seed}.json"
    if not path.is_file():
        raise HTTPException(404, f"no rollout for baseline {name!r} seed {seed}")
    return FileResponse(path, media_type="application/json")


@app.post("/baselines/compute")
def compute_baselines_route(req: BaselinesRequest) -> dict[str, Any]:
    if FAKE:
        return {"ok": True, "fake": True, "baselines": {}}
    from .eval.baselines import compute_baselines

    return {"ok": True, "suite": req.suite,
            "baselines": compute_baselines(store, suite_name=req.suite, n_episodes=req.n_episodes, names=req.names)}


@app.post("/runs/compare")
def compare_runs(req: CompareRequest) -> dict[str, Any]:
    missing = [r for r in req.run_ids if not store.exists(r)]
    if missing:
        raise HTTPException(404, f"unknown runs: {missing}")
    suites = {r: store.run_suite(r) for r in req.run_ids}
    reports = {r: store.benchmark(r, suites[r]) for r in req.run_ids}
    gates_table: dict[str, dict[str, Any]] = {}
    for rid, rep in reports.items():
        for g in (rep or {}).get("gates", []):
            gates_table.setdefault(g["gate"], {})[rid] = g["value"]
    configs = {r: TrainConfig.model_validate(store.config(r)).resolved().model_dump(mode="json") for r in req.run_ids}
    first = req.run_ids[0]
    diffs = {r: config_diff(configs[first], configs[r]) for r in req.run_ids[1:]}
    return {"runs": {r: store.summary(r) for r in req.run_ids}, "suites": suites,
            "cross_suite": len(set(suites.values())) > 1, "gates_table": gates_table,
            "config_diffs_vs_first": diffs, "reflections": {r: (rep or {}).get("reflection") for r, rep in reports.items()}}


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str) -> dict[str, Any]:
    d = store.run_dir(run_id)
    if not d.is_dir():
        raise HTTPException(404, f"run {run_id!r} not found")
    return {"files": sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())}


@app.get("/runs/{run_id}/artifacts/{name:path}")
def get_artifact(run_id: str, name: str) -> Any:
    d = store.run_dir(run_id).resolve()
    p = (d / name).resolve()
    if d not in p.parents or not p.is_file():
        raise HTTPException(404, "no such artifact")
    return FileResponse(p)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("TRAINER_HOST", "0.0.0.0"), port=int(os.getenv("TRAINER_PORT", "8009")))


if __name__ == "__main__":
    main()
