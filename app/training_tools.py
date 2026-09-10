"""Training tools for the GLM harness: schemas + bodies.

The LLM never sees training code; it edits a validated JSON config, launches
a run, reads the gate report + reflection, and proposes the next patch.
``bittle_train_and_benchmark`` is the "one tool call = one cycle" composite.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .trainer_client import TrainerClient, TrainerError, TrainerUnavailable

TRAINING_TOOL_NAMES = (
    "bittle_list_tasks",
    "bittle_propose_config",
    "bittle_train_policy",
    "bittle_train_status",
    "bittle_benchmark_policy",
    "bittle_train_and_benchmark",
    "bittle_list_runs",
    "bittle_compare_runs",
    "bittle_replay_rollout",
    "bittle_trainer_health",
    "bittle_diagnose_run",
)

_PATCH = {
    "type": "object",
    "description": (
        "Partial TrainConfig to merge onto the base (defaults or base_run_id). Keys: reward.weights.{tracking_lin_vel,"
        "tracking_ang_vel,lin_vel_z,ang_vel_xy,orientation,base_height,action_rate,energy,joint_saturation,feet_air_time,"
        "feet_slip,stand_still,foot_clearance,stumble,slope_progress,stall}, reward.tracking_sigma, curriculum_stage (0-2), "
        "terrain.{level (0 flat,1 slope,2 rocks,3 rocks+slope),slope_deg,n_boxes,box_height_m,box_size_m,spawn_jitter_m}, "
        "commands.{vx,vy,wz,resample_s}, "
        "dr.{enabled,friction,mass_scale,payload_g,kp,forcerange,damping,frictionloss,latency_steps,gyro_noise,...}, "
        "ppo.{num_envs,num_timesteps,learning_rate,entropy_cost,discounting,unroll_length,num_minibatches,batch_size,"
        "policy_hidden,value_hidden,num_evals}, episode_seconds, control_hz (25|50), action_scale_deg, "
        "obs.{history_n,phase_clock}, seed. Unknown keys are rejected."
    ),
}

TRAINING_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "bittle_trainer_health",
        "description": "Check the RL trainer service (GPU box): reachable, physics backend, active jobs.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_list_tasks",
        "description": "The task catalogue: every trainable task with its goal, aliases, gate suite, prerequisites, the config keys that matter, whether its prerequisite is satisfied (and by which run) and the run to warm-start from. Call this FIRST and match the operator's words to a task; the task decides the gate suite.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_propose_config",
        "description": "Validate a training config patch WITHOUT running it. Returns the resolved config, the diff vs the base, errors and warnings. Call with an empty patch to see the defaults and the editable keys.",
        "parameters": {"type": "object", "properties": {
            "config_patch": _PATCH,
            "base_run_id": {"type": "string", "description": "Start from this run's config instead of the defaults"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_train_policy",
        "description": "Submit an RL training run (PPO in MuJoCo Warp on the GPU box) with a config patch. Returns a run_id immediately; poll with bittle_train_status. Prefer bittle_train_and_benchmark for a full cycle.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "config_patch": _PATCH, "base_run_id": {"type": "string"},
            "notes": {"type": "string", "description": "Why this config (hypothesis)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_train_status",
        "description": "Status / progress / reward curve of a run. wait_s (<=300) long-polls until it is trained or done.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "wait_s": {"type": "number", "description": "0-300 seconds"}},
            "required": ["run_id"]}}},
    {"type": "function", "function": {
        "name": "bittle_benchmark_policy",
        "description": "Run a gate suite (the policy test suite) on a trained run in CPU MuJoCo. Omit `suite` to use the run's own suite (its task's). Another suite is an ablation and needs force=true; its numbers never enter the leaderboard. Returns gate table, score and a reflection string. dr_sweep adds robustness presets (slower).",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "suite": {"type": "string", "description": "omit = the run's own suite"},
            "force": {"type": "boolean", "description": "required to benchmark on a suite other than the run's own"},
            "dr_sweep": {"type": "boolean"}, "dual_sim": {"type": "boolean"},
            "wait_s": {"type": "number", "description": "0-300 seconds to wait for the report"}},
            "required": ["run_id"]}}},
    {"type": "function", "function": {
        "name": "bittle_train_and_benchmark",
        "description": "ONE FULL CYCLE: submit training for a task with a config patch, wait for it to finish, benchmark it on THAT TASK'S gate suite, and return {run_id, task, suite, gates, score, reflection, config_diff, curve}. This is the tool to use for iterative training. Takes minutes; progress is streamed.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "task": {"type": "string", "description": "Task name from bittle_list_tasks (flat_walk, slope_up, rough_walk, ...). Decides the gate suite. Omit to keep the parent run's task."},
            "config_patch": _PATCH, "base_run_id": {"type": "string"},
            "notes": {"type": "string", "description": "Hypothesis for this change"},
            "dr_sweep": {"type": "boolean"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_diagnose_run",
        "description": "Why did a run's gates fail? Returns each gate with the run's value, the parent run's value and the firmware-trot baseline, plus the reward-term breakdown (share of total reward per term). A term below ~1% share cannot steer training: multiply its weight by 10-100x, not 2x.",
        "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_list_runs",
        "description": "Leaderboard of runs (score, gates passed, distance, fall rate) plus the OpenCat baseline gaits. Scores are only comparable WITHIN a suite: pass `suite` to rank one task's runs; without it you get best_by_suite.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer"}, "sort": {"type": "string", "enum": ["score", "created"]},
            "suite": {"type": "string", "description": "e.g. flat_v1, slope_v1, rough_v1"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_compare_runs",
        "description": "Side-by-side gate values and config diffs for 2-10 runs.",
        "parameters": {"type": "object", "properties": {
            "run_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["run_ids"]}}},
    {"type": "function", "function": {
        "name": "bittle_replay_rollout",
        "description": "Play a benchmark rollout of a run (or a baseline gait) in the 3D viewer. Returns the summary and a viewer URL; frames never enter the chat.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}, "seed": {"type": "integer"},
            "source": {"type": "string", "description": "'benchmark' (default) or 'baseline:<name>' e.g. baseline:opencat_trF"}},
            "required": []}}},
]


def _err(kind: str, detail: Any) -> dict[str, Any]:
    return {"ok": False, "error": kind, "detail": str(detail)[:600]}


def _compact_report(rep: dict[str, Any]) -> dict[str, Any]:
    gates = [{"gate": g["gate"], "value": g["value"], "op": g["op"], "threshold": g["threshold"], "pass": g["pass"],
              **({"note": g["note"]} if g.get("note") else {})} for g in rep.get("gates", [])]
    keep = ("run_id", "task", "suite", "suite_version", "gates_passed", "gates_total", "passed", "score", "reflection", "n_episodes")
    out = {k: rep[k] for k in keep if k in rep}
    out["gates"] = gates
    m = rep.get("metrics", {})
    out["metrics"] = {k: m[k] for k in ("fall_rate", "forward_distance_p50", "vel_tracking_rmse", "heading_yaw_deg",
                                        "lateral_drift_m", "joint_saturation_pct", "action_smoothness_deg",
                                        "mean_tilt_deg", "energy_proxy_w", "first_fall_time_mean",
                                        "climb_height_p50", "stumble_rate", "foot_clearance_p50_mm",
                                        "peak_joint_speed_rad_s", "stall_fraction", "stall_concurrent_max") if k in m}
    ctx = rep.get("context") or {}
    if ctx:
        out["reward_shares_pct"] = ctx.get("reward_shares_pct")
        out["parent_run_id"] = ctx.get("parent_run_id")
        for k in ("parent_note", "baseline_note"):
            if ctx.get(k):
                out[k] = ctx[k]
    return out


class TrainingTools:
    """Bodies of the training tools. ``progress`` receives keepalive events for long cycles."""

    def __init__(self, client: TrainerClient, settings, *, progress=None):
        self.client = client
        self.settings = settings
        self.progress = progress
        self.last_run_id: str | None = None
        self.last_progress: dict[str, Any] | None = None

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.client.configured:
            return _err("trainer_not_configured", "Set BITTLE_TRAINER_URL to the trainer service (GPU box :8009)")
        try:
            handler = getattr(self, "_" + name.removeprefix("bittle_"))
            return await handler(args)
        except TrainerUnavailable as exc:
            return _err("trainer_unavailable", exc)
        except TrainerError as exc:
            return {"ok": False, "error": "trainer_error", "status": exc.status, "detail": exc.detail}

    # ── tools ──────────────────────────────────────────────────────────
    async def _trainer_health(self, args):
        return {"ok": True, "health": await self.client.health()}

    async def _list_tasks(self, args):
        res = await self.client.tasks()
        return {"ok": True, **res, "hint": "pick the task whose goal/aliases match the operator's words; if none matches, say so"}

    async def _propose_config(self, args):
        res = await self.client.validate(args.get("config_patch") or {}, args.get("base_run_id"))
        return {"ok": True, "config_hash": res["config_hash"], "diff": res["diff"], "warnings": res.get("warnings", []),
                "resolved": res["resolved"]}

    @staticmethod
    def _task_and_patch(args) -> tuple[str | None, dict[str, Any]] | dict[str, Any]:
        """The task rides in the patch; a task that disagrees with config_patch.task is refused here,
        before anything is submitted (a slope run silently gated on flat is the failure this prevents)."""
        patch = dict(args.get("config_patch") or {})
        task = args.get("task") or None
        if task and patch.get("task") not in (None, task):
            return _err("task_conflict", f"task={task!r} but config_patch.task={patch['task']!r}")
        return task, patch

    async def _train_policy(self, args):
        tp = self._task_and_patch(args)
        if isinstance(tp, dict):
            return tp
        task, patch = tp
        res = await self.client.submit(patch, name=args.get("name", ""), base_run_id=args.get("base_run_id"),
                                       notes=args.get("notes", ""), task=task)
        self.last_run_id = res["run_id"]
        return {"ok": True, **res, "hint": "poll bittle_train_status(run_id, wait_s=300)"}

    async def _train_status(self, args):
        wait = float(min(max(args.get("wait_s", 0) or 0, 0), 300))
        st = await self.client.status(args["run_id"], wait_s=wait)
        return {"ok": True, **_compact_state(st)}

    async def _benchmark_policy(self, args):
        wait = float(min(max(args.get("wait_s", 0) or 0, 0), 300))
        res = await self.client.benchmark(args["run_id"], suite=args.get("suite") or None, force=bool(args.get("force")),
                                          dr_sweep=bool(args.get("dr_sweep")), dual_sim=bool(args.get("dual_sim")),
                                          wait_s=wait)
        if "gates" in res:
            return {"ok": True, **_compact_report(res)}
        return {"ok": True, **res, "hint": "report not ready; call bittle_benchmark_policy again with wait_s"}

    async def _train_and_benchmark(self, args):
        t0 = time.monotonic()
        tp = self._task_and_patch(args)
        if isinstance(tp, dict):
            return tp
        task, patch = tp
        sub = await self.client.submit(patch, name=args.get("name", ""), base_run_id=args.get("base_run_id"),
                                       notes=args.get("notes", ""), task=task)
        run_id = sub["run_id"]
        self.last_run_id = run_id
        budget = float(getattr(self.settings, "trainer_cycle_timeout", 1800.0))
        st = await self._wait(run_id, until="trained", budget=budget, t0=t0)
        if st.get("status") not in ("trained", "done"):
            return {"ok": False, "error": "training_failed", "run_id": run_id, "status": st.get("status"),
                    "detail": st.get("error"), "config_diff": sub.get("diff")}
        # no suite: the trainer benchmarks on the run's OWN suite (its task's)
        await self.client.benchmark(run_id, dr_sweep=bool(args.get("dr_sweep")), wait_s=0)
        st = await self._wait(run_id, until="done", budget=budget, t0=t0)
        rep = None
        try:
            rep = await self.client.get_benchmark(run_id)
        except TrainerError:
            pass
        if not rep:
            return {"ok": False, "error": "benchmark_failed", "run_id": run_id, "status": st.get("status"),
                    "detail": st.get("error"), "config_diff": sub.get("diff")}
        return {"ok": True, "run_id": run_id, "task": sub.get("task"), "suite": sub.get("suite"),
                "config_diff": sub.get("diff"), "training": _compact_state(st).get("training"),
                **_compact_report(rep), "elapsed_s": round(time.monotonic() - t0, 1)}

    async def _wait(self, run_id: str, *, until: str, budget: float, t0: float) -> dict[str, Any]:
        while True:
            remaining = budget - (time.monotonic() - t0)
            if remaining <= 0:
                st = await self.client.status(run_id)
                st["error"] = st.get("error") or f"cycle budget of {budget:.0f}s exhausted while {st.get('status')}"
                return st
            st = await self.client.status(run_id, wait_s=min(15.0, remaining), until=until)
            status = st.get("status")
            self.last_progress = {"run_id": run_id, "status": status, "progress": st.get("progress"),
                                  "elapsed_s": round(time.monotonic() - t0, 1)}
            if self.progress:
                await self.progress(self.last_progress)
            terminal = {"failed", "cancelled"}
            if until == "trained" and status in ({"trained", "done", "benchmarking"} | terminal):
                return st
            if until == "done" and status in ({"done"} | terminal):
                return st
            if until == "done" and status == "trained" and st.get("error"):
                return st
            await asyncio.sleep(0)

    async def _diagnose_run(self, args):
        run_id = args.get("run_id") or self.last_run_id
        if not run_id:
            return _err("missing_run_id", "give a run_id or train first")
        rep = await self.client.get_benchmark(run_id)
        ctx = rep.get("context") or {}
        rows = []
        for g in rep.get("gates", []):
            if g["value"] is None:
                continue
            pg = ctx.get("per_gate", {}).get(g["gate"], {})
            rows.append({"gate": g["gate"], "pass": g["pass"], "value": g["value"], "threshold": g["threshold"], "op": g["op"],
                         "parent": pg.get("parent"), "baseline_trot": pg.get("baseline_trot"),
                         "reward_term": pg.get("term"), "term_share_pct": pg.get("term_share_pct")})
        shares = ctx.get("reward_shares_pct") or {}
        weights = ctx.get("reward_weights") or {}
        off = sorted(k for k, w in weights.items() if float(w) == 0.0 and any(r["reward_term"] == k and r["pass"] is False for r in rows))
        weak = [k for k, v in shares.items() if v is not None and v < 1.0 and k not in off]
        notes = [ctx[k] for k in ("parent_note", "baseline_note") if ctx.get(k)]
        bench_d = (rep.get("metrics") or {}).get("forward_distance_p50")
        train_vs_bench = {k.replace("train_", ""): ctx[k] for k in ("train_distance_x", "train_bar_distance_x", "train_bar_distance_p50", "train_bar_fall_rate") if ctx.get(k) is not None}
        if bench_d is not None:
            train_vs_bench["bench_distance_p50"] = bench_d
            if ctx.get("train_distance_x") is not None:
                train_vs_bench["train_over_bench_ratio"] = round(float(ctx["train_distance_x"]) / max(float(bench_d), 1e-6), 2)
        gap = ctx.get("terrain_gap") or {}
        mismatch = bool(gap.get("easier_than_protocol")) and train_vs_bench.get("train_over_bench_ratio", 0) > 1.5
        if mismatch:
            advice = ("TRAIN/BENCH MISMATCH: the run trains on an easier field than the suite protocol (see terrain_gap) and its "
                      "training curve walks far more than the benchmark. Raise terrain.box_height_m / terrain.n_boxes (or slope_deg) "
                      "to cover the protocol first; reward weights cannot fix a field the optimiser never sees. A terrain or "
                      "gait-shaping change needs ppo.num_timesteps 30M, not 10M.")
        elif off:
            advice = (f"The failing gates' reward terms are switched OFF ({', '.join(off)} = 0.0): set those weights first "
                      "(the task's config_patch carries defaults), then train 30M -- a gait has to be reshaped.")
        elif weak:
            advice = (f"Terms with <1% reward share ({', '.join(weak)}) cannot steer the policy; a gate tied to one of them "
                      "needs a 10-100x weight change. 10M warm steps suffice for a weight tweak; 30M when the gait must change.")
        else:
            advice = ("All reward terms carry weight. If a gate did not move vs the parent, the weight is not the lever: "
                      "raise ppo.num_timesteps to 30M or the terrain difficulty; adjust thresholds' neighbours by 2-5x otherwise.")
        stuck = {k: (rep.get("metrics") or {}).get(k) for k in ("stuck_episode_rate", "stuck_seconds_p50", "stuck_x_p50", "stuck_limb_share")
                 if (rep.get("metrics") or {}).get(k) is not None}
        return {"ok": True, "run_id": run_id, "task": rep.get("task"), "suite": rep.get("suite"),
                "suite_version": rep.get("suite_version"), "gates": rows,
                "reward_shares_pct": shares, "parent_run_id": ctx.get("parent_run_id"),
                **({"reward_terms_off": off} if off else {}),
                **({"train_vs_bench": train_vs_bench} if train_vs_bench else {}),
                **({"terrain_gap": gap} if gap else {}),
                **({"stuck": stuck} if stuck else {}),
                **({"notes": notes} if notes else {}),
                "advice": advice,
                "reflection": rep.get("reflection")}

    async def _list_runs(self, args):
        res = await self.client.list_runs(sort=args.get("sort", "score"), limit=int(args.get("limit", 10) or 10),
                                          suite=args.get("suite") or None)
        return {"ok": True, **res}

    async def _compare_runs(self, args):
        return {"ok": True, **await self.client.compare(list(args.get("run_ids") or []))}

    async def _replay_rollout(self, args):
        seed = int(args.get("seed", 0) or 0)
        source = str(args.get("source") or "benchmark")
        if source.startswith("baseline:"):
            name = source.split(":", 1)[1]
            base = (await self.client.baselines()).get(name)
            if not base:
                return _err("not_found", f"no baseline {name!r}")
            return {"ok": True, "source": source, "summary": {"score": base.get("score"), "reflection": base.get("reflection")},
                    "viewer_url": f"/api/training/baselines/{name}/rollout?seed={seed}"}
        run_id = args.get("run_id") or self.last_run_id
        if not run_id:
            return _err("missing_run_id", "give a run_id or train first")
        rep = await self.client.get_benchmark(run_id)
        return {"ok": True, "run_id": run_id, "seed": seed,
                "summary": {"score": rep.get("score"), "gates_passed": rep.get("gates_passed"),
                            "gates_total": rep.get("gates_total"), "reflection": rep.get("reflection")},
                "viewer_url": f"/api/training/rollout/{run_id}?seed={seed}"}


def _compact_state(st: dict[str, Any]) -> dict[str, Any]:
    out = {
        "run_id": st.get("run_id"), "name": st.get("name"), "status": st.get("status"), "error": st.get("error"),
        "progress": st.get("progress"), "config_hash": st.get("config_hash"), "parent": st.get("parent"),
        "training": {k: st.get("metrics", {}).get(k) for k in ("reward_first", "reward_final", "distance_final",
                                                                "elapsed_s", "steps_per_s_mean", "impl")},
        "curve": [{"step": p.get("step"), "reward": p.get("reward"), "distance_x": p.get("distance_x")}
                  for p in st.get("curve", [])][-8:],
    }
    if st.get("benchmark"):
        out["benchmark"] = _compact_report(st["benchmark"])
    return out
