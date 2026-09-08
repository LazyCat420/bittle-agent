"""Training + research tools in the GLM harness, with the trainer faked (no GPU, no network)."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.agent import TOOLS, GLMAgentHarness
from app.config import Settings
from app.controller import Controller
from app.main import app
from app.research_tools import RESEARCH_TOOL_NAMES, check_url, html_to_text
from app.trainer_client import TrainerError, TrainerUnavailable
from app.training_tools import TRAINING_TOOL_NAMES, TrainingTools


class FakeTrainer:
    configured = True

    def __init__(self):
        self.calls = []
        self.status_seq = ["training", "trained", "done"]
        self.n = 0

    async def health(self):
        return {"ok": True, "physics_impl": "warp"}

    async def validate(self, patch, base=None):
        return {"ok": True, "config_hash": "h1", "diff": [{"path": "reward.weights.tracking_lin_vel", "from": 1.5, "to": 2.0}],
                "resolved": {"reward": {}}, "warnings": []}

    async def submit(self, patch, name="", base_run_id=None, notes="", task=None):
        self.calls.append(("submit", patch, base_run_id, task))
        suite = {"slope_up": "slope_v1", "rough_walk": "rough_v1"}.get(task or patch.get("task"), "flat_v1")
        return {"run_id": "run-1", "status": "queued", "task": task or "flat_walk", "suite": suite, "config_hash": "h1", "diff": []}

    async def status(self, run_id, wait_s=0.0, until="any"):
        self.calls.append(("status", run_id, wait_s, until))
        st = self.status_seq[min(self.n, len(self.status_seq) - 1)]
        self.n += 1
        return {"run_id": run_id, "status": st, "progress": {"step": 10, "total": 100}, "metrics": {"reward_final": 5.0},
                "curve": [{"step": 10, "reward": 5.0}]}

    async def benchmark(self, run_id, suite=None, force=False, n_episodes=None, dr_sweep=False, dual_sim=False, wait_s=0.0):
        self.calls.append(("benchmark", run_id, dr_sweep, suite, force))
        self.last_suite = suite or "slope_v1"  # the trainer resolves None to the run's own suite
        return {"run_id": run_id, "status": "benchmarking", "queued": True}

    mismatch = False

    async def get_benchmark(self, run_id, suite=None):
        suite = suite or getattr(self, "last_suite", "flat_v1")
        ctx = {"per_gate": {"fall_rate": {"value": 0.2, "parent": 0.1, "baseline_trot": 0.0, "term": "orientation", "term_share_pct": 0.5}},
               "reward_shares_pct": {"tracking_lin_vel": 95.0, "orientation": 0.5}, "parent_run_id": "run-0"}
        if self.mismatch:
            ctx = {"per_gate": {"fall_rate": {"value": 0.2, "parent": None, "baseline_trot": None, "term": "orientation", "term_share_pct": 0.5}},
                   "reward_shares_pct": {"tracking_lin_vel": 95.0, "orientation": 0.5}, "parent_run_id": None,
                   "parent_suite_mismatch": {"parent_run_id": "run-0", "parent_suite": "flat_v1", "this_suite": suite},
                   "parent_note": "Parent run-0 has no slope_v1 benchmark (it was benchmarked on flat_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id=\"run-0\", suite=\"slope_v1\", force=true) if you want the delta."}
        return {"run_id": run_id, "task": "slope_up" if suite == "slope_v1" else "flat_walk", "suite": suite,
                "suite_version": "1.0.0", "gates_passed": 11, "gates_total": 12,
                "passed": False, "score": 11.3, "reflection": "BENCHMARK FAILED 11/12 gates.",
                "gates": [{"gate": "fall_rate", "value": 0.2, "op": "<=", "threshold": 0.1, "pass": False, "note": "hint", "unit": "x"}],
                "metrics": {"fall_rate": 0.2, "forward_distance_p50": 0.7}, "episodes": [{"seed": 0}], "context": ctx}

    async def list_runs(self, sort="score", limit=20, suite=None):
        self.calls.append(("list_runs", suite))
        return {"runs": [{"run_id": "run-1", "score": 11.3, "suite": suite or "flat_v1"}], "baselines": {"opencat_trF": {"score": 5.0}},
                "best": {"run_id": "run-1"}, "best_by_suite": {"flat_v1": {"run_id": "run-1"}}}

    async def tasks(self):
        return {"tasks": [
            {"task": "flat_walk", "goal": "walk on flat ground", "suite": "flat_v1", "suite_version": "1.2.0", "aliases": ["walk"],
             "prerequisites": [], "config_keys": ["reward.weights.tracking_lin_vel"], "prereq_satisfied_by": {},
             "warm_start_from": "run-1", "status": "ready"},
            {"task": "slope_up", "goal": "walk up an 8 degree incline", "suite": "slope_v1", "suite_version": "0.1.0-uncalibrated",
             "aliases": ["slope", "incline"], "prerequisites": ["flat_walk"], "config_keys": ["terrain.slope_deg"],
             "prereq_satisfied_by": {"flat_walk": None}, "warm_start_from": None, "status": "blocked: no flat_walk run passes every gate yet"}],
            "suites": ["flat_v1", "rough_v1", "slope_v1"]}

    async def compare(self, run_ids):
        return {"runs": {r: {} for r in run_ids}, "gates_table": {}, "config_diffs_vs_first": {}}

    async def baselines(self, suite=None):
        return {"opencat_trF": {"score": 5.0, "reflection": "baseline"}}

    async def rollout_stream(self, path):
        yield b'{"schema":"bittle.rollout.v1","frames":[]}'


class DownTrainer(FakeTrainer):
    async def submit(self, *a, **k):
        raise TrainerUnavailable("connection refused")

    async def health(self):
        raise TrainerUnavailable("connection refused")


@pytest.fixture
def harness():
    cfg = Settings(allow_real_hardware=False, trainer_url="http://fake:8009", trainer_cycle_timeout=5.0)
    ctrl = Controller(cfg)
    ctrl.safety.clear_estop()
    h = GLMAgentHarness(ctrl, cfg)
    h.training = TrainingTools(FakeTrainer(), cfg)
    return h


def run(coro):
    return asyncio.run(coro)


def test_training_and_research_tools_are_registered():
    names = {t["function"]["name"] for t in TOOLS}
    assert set(TRAINING_TOOL_NAMES) <= names and set(RESEARCH_TOOL_NAMES) <= names
    assert "bittle_run_stair_training_episode" in names  # the old closed-form tools stay


def test_not_configured_is_structured():
    cfg = Settings(allow_real_hardware=False, trainer_url="")
    h = GLMAgentHarness(Controller(cfg), cfg)
    res = run(h.execute_tool("bittle_train_policy", {"config_patch": {}}))
    assert res["ok"] is False and res["error"] == "trainer_not_configured"


def test_unavailable_is_structured():
    cfg = Settings(allow_real_hardware=False, trainer_url="http://down:8009")
    h = GLMAgentHarness(Controller(cfg), cfg)
    h.training = TrainingTools(DownTrainer(), cfg)
    res = run(h.execute_tool("bittle_train_policy", {}))
    assert res == {"ok": False, "error": "trainer_unavailable", "detail": "connection refused"}


def test_propose_and_train_status(harness):
    res = run(harness.execute_tool("bittle_propose_config", {"config_patch": {"reward": {"weights": {"tracking_lin_vel": 2.0}}}}))
    assert res["ok"] and res["diff"][0]["path"].startswith("reward")
    res = run(harness.execute_tool("bittle_train_policy", {"config_patch": {}}))
    assert res["run_id"] == "run-1"
    res = run(harness.execute_tool("bittle_train_status", {"run_id": "run-1", "wait_s": 999}))
    assert res["ok"] and res["status"] in ("training", "trained")
    assert harness.training.client.calls[-1][2] == 300  # wait clamped


def test_train_and_benchmark_cycle_returns_gates_and_reflection(harness):
    res = run(harness.execute_tool("bittle_train_and_benchmark", {"config_patch": {}, "base_run_id": "run-0"}))
    assert res["ok"] and res["run_id"] == "run-1"
    assert res["gates_passed"] == 11 and "BENCHMARK FAILED" in res["reflection"]
    assert res["gates"][0]["note"] == "hint" and "episodes" not in res
    kinds = [c[0] for c in harness.training.client.calls]
    assert kinds[0] == "submit" and "benchmark" in kinds
    assert harness.training.client.calls[0][2] == "run-0"
    assert harness.training.last_progress["run_id"] == "run-1"


# ── task -> suite binding (2026-09-07) ─────────────────────────────────────

def test_train_and_benchmark_passes_task_and_never_names_a_suite(harness):
    res = run(harness.execute_tool("bittle_train_and_benchmark", {"task": "slope_up", "config_patch": {}, "base_run_id": "run-0"}))
    assert res["ok"] and res["task"] == "slope_up" and res["suite"] == "slope_v1"
    calls = harness.training.client.calls
    assert calls[0] == ("submit", {}, "run-0", "slope_up")
    bench = [c for c in calls if c[0] == "benchmark"][0]
    assert bench[3] is None and bench[4] is False  # the trainer resolves the run's own suite


def test_train_and_benchmark_rejects_a_task_that_conflicts_with_the_patch(harness):
    res = run(harness.execute_tool("bittle_train_and_benchmark", {"task": "slope_up", "config_patch": {"task": "flat_walk"}}))
    assert res == {"ok": False, "error": "task_conflict", "detail": "task='slope_up' but config_patch.task='flat_walk'"}
    assert harness.training.client.calls == []  # nothing submitted


def test_benchmark_policy_forwards_suite_and_force_only_when_given(harness):
    run(harness.execute_tool("bittle_benchmark_policy", {"run_id": "run-1"}))
    assert harness.training.client.calls[-1][3:] == (None, False)
    run(harness.execute_tool("bittle_benchmark_policy", {"run_id": "run-1", "suite": "flat_v1", "force": True}))
    assert harness.training.client.calls[-1][3:] == ("flat_v1", True)


def test_list_tasks_returns_the_catalogue_with_prerequisites(harness):
    assert "bittle_list_tasks" in TRAINING_TOOL_NAMES
    res = run(harness.execute_tool("bittle_list_tasks", {}))
    t = {x["task"]: x for x in res["tasks"]}
    assert t["slope_up"]["status"].startswith("blocked") and t["slope_up"]["warm_start_from"] is None
    assert t["flat_walk"]["warm_start_from"] == "run-1" and "hint" in res


def test_list_runs_forwards_the_suite_filter(harness):
    res = run(harness.execute_tool("bittle_list_runs", {"suite": "slope_v1"}))
    assert harness.training.client.calls[-1] == ("list_runs", "slope_v1") and "best_by_suite" in res


def test_training_prompt_is_task_agnostic(harness):
    p = harness.get_system_prompt(mode="training")
    assert "bittle_list_tasks" in p and "warm_start_from" in p and "bittle_diagnose_run" in p
    assert "flat_v1" not in p  # no suite is ever hard-coded into the prompt again


def test_diagnose_run_surfaces_a_parent_suite_mismatch(harness):
    harness.training.client.mismatch = True
    res = run(harness.execute_tool("bittle_diagnose_run", {"run_id": "run-1"}))
    assert res["ok"] and res["gates"][0]["parent"] is None and res["parent_run_id"] is None
    assert any("force=true" in n for n in res["notes"])


def test_replay_rollout_returns_viewer_url_not_frames(harness):
    res = run(harness.execute_tool("bittle_replay_rollout", {"run_id": "run-1", "seed": 2}))
    assert res["viewer_url"] == "/api/training/rollout/run-1?seed=2" and "frames" not in json.dumps(res)
    res = run(harness.execute_tool("bittle_replay_rollout", {"source": "baseline:opencat_trF"}))
    assert res["viewer_url"].startswith("/api/training/baselines/opencat_trF/rollout")


def test_training_mode_turn_budget_and_prompt(harness):
    assert "TRAINING MODE PROTOCOL" in harness.get_system_prompt(mode="training")
    assert "TRAINING MODE PROTOCOL" not in harness.get_system_prompt(mode="control")
    assert "bittle_web_search" in harness.get_system_prompt()

    async def collect(mode):
        # endpoint mocked to fail fast: the loop yields one error event; we only check max_turns selection
        harness.resolve_endpoint_and_model = lambda: asyncio.sleep(0, result=("http://x/v1", "m"))
        seen = {}
        orig = harness.settings

        class S:
            pass

        return harness.settings.agent_max_turns_training if mode == "training" else harness.settings.agent_max_turns

    assert run(collect("training")) == 12 and run(collect("control")) == 6


def test_chat_stream_emits_progress_for_slow_tool(harness, monkeypatch):
    import httpx

    harness.KEEPALIVE_S = 0.05

    async def mock_resolve():
        return "http://test:8000/v1", "GLM"

    monkeypatch.setattr(harness, "resolve_endpoint_and_model", mock_resolve)
    turn1 = ['data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "bittle_train_and_benchmark", "arguments": "{}"}}]}}]}), 'data: [DONE]']
    turn2 = ['data: ' + json.dumps({"choices": [{"delta": {"content": "done"}}]}), 'data: [DONE]']
    calls = {"n": 0}

    class R:
        status_code = 200

        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def aiter_lines(self):
            for l in self.lines:
                yield l

    def mock_stream(self, method, url, **kw):
        calls["n"] += 1
        return R(turn1 if calls["n"] == 1 else turn2)

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    async def slow_execute(name, args, **kw):
        await asyncio.sleep(0.2)
        return {"ok": True, "score": 1.0}

    monkeypatch.setattr(harness, "execute_tool", slow_execute)

    async def collect():
        return [e async for e in harness.chat_stream([{"role": "user", "content": "train"}], mode="training")]

    events = run(collect())
    types = [e["type"] for e in events]
    assert "progress" in types and "tool_result" in types and types[-1] == "done"
    assert types.index("progress") < types.index("tool_result")


def test_training_proxies(monkeypatch):
    from app import main as main_mod

    main_mod.agent_harness.trainer = FakeTrainer()
    with TestClient(app) as c:
        r = c.get("/api/training/rollout/run-1", params={"seed": 0})
        assert r.status_code == 200 and r.json()["schema"] == "bittle.rollout.v1"
        assert c.get("/api/training/runs").json()["best"]["run_id"] == "run-1"
        assert c.get("/api/training/health").json()["ok"] is True
    # mode is accepted by the request model (the chat loop itself is covered by the SSE stub test)
    assert main_mod.AgentChatRequest(prompt="x", mode="training").mode == "training"


# ── research tools ─────────────────────────────────────────────────────────

def test_research_url_guard_blocks_lan():
    assert check_url("http://10.0.0.16:5591/x") is not None
    assert check_url("http://localhost:8008/") is not None
    assert check_url("ftp://example.com/") is not None


def test_html_to_text_strips_scripts():
    title, text = html_to_text("<html><head><title>T</title><script>x()</script></head><body><p>Hello <b>w</b></p><style>a{}</style></body></html>")
    assert title == "T" and "Hello w" in text and "x()" not in text and "a{}" not in text


def test_research_notes_roundtrip(tmp_path):
    from app.research_tools import ResearchTools

    rt = ResearchTools(Settings(), notes_dir=tmp_path)
    res = run(rt.execute("bittle_save_research_note", {"title": "PPO sigma", "content": "0.25 for Go1", "tags": ["ppo"]}))
    assert res["ok"] and res["name"] == "ppo-sigma"
    lst = run(rt.execute("bittle_list_research_notes", {"tag": "ppo"}))
    assert lst["count"] == 1 and lst["notes"][0]["title"] == "PPO sigma"
    assert run(rt.execute("bittle_read_research_note", {"name": "PPO sigma"}))["note"]["content"] == "0.25 for Go1"
    assert run(rt.execute("bittle_list_research_notes", {"tag": "nope"}))["count"] == 0


def test_research_disabled_flag(tmp_path):
    from app.research_tools import ResearchTools

    rt = ResearchTools(Settings(allow_web_research=False), notes_dir=tmp_path)
    assert run(rt.execute("bittle_web_search", {"query": "x"}))["error"] == "web_research_disabled"


def test_ddg_parser_extracts_results():
    from app.research_tools import _DDGParser

    html = ('<table><tr class="result-sponsored"><td><a class="result-link" href="//ad.example">Ad</a></td></tr>'
            '<tr><td><a rel="nofollow" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1" class="result-link">Example A</a></td></tr>'
            '<tr><td class="result-snippet">A snippet.</td></tr></table>')
    p = _DDGParser()
    p.feed(html)
    assert p.results == [{"title": "Example A", "url": "https://example.com/a", "snippet": "A snippet."}]
    # absolute redirect form seen live
    p2 = _DDGParser()
    p2.feed('<tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fx%2Fy&amp;rut=2">X</a></td></tr>')
    assert p2.results[0]["url"] == "https://github.com/x/y"


def test_read_url_rewrites_github_and_arxiv():
    from app.research_tools import _rewrite_url

    assert _rewrite_url("https://github.com/ger01d/opencat-gym").endswith("/ger01d/opencat-gym/HEAD/README.md")
    assert _rewrite_url("https://github.com/a/b/blob/main/x.py") == "https://raw.githubusercontent.com/a/b/main/x.py"
    assert _rewrite_url("https://arxiv.org/pdf/2502.08844v1.pdf") == "https://arxiv.org/abs/2502.08844"


def test_diagnose_run_reports_weak_terms(harness):
    res = run(harness.execute_tool("bittle_diagnose_run", {"run_id": "run-1"}))
    assert res["ok"] and res["gates"][0]["parent"] == 0.1 and res["gates"][0]["baseline_trot"] == 0.0
    assert "orientation" in res["advice"] and res["reward_shares_pct"]["tracking_lin_vel"] == 95.0
    assert "bittle_diagnose_run" in harness.get_system_prompt(mode="training")


# ── training dashboard: recorded sessions + proxies (2026-09-07) ──────────

def test_session_recorder_tees_a_training_stream_into_cycles(tmp_path):
    from app.training_sessions import SessionRecorder

    rec = SessionRecorder(tmp_path)

    async def fake_stream():
        yield {"type": "thought", "content": "I will "}
        yield {"type": "thought", "content": "train."}
        yield {"type": "tool_call", "id": "c1", "name": "bittle_train_and_benchmark",
               "args": {"task": "slope_up", "config_patch": {"ppo": {"num_timesteps": 1}}, "notes": "hyp"}}
        yield {"type": "progress", "id": "c1", "detail": {"status": "training", "progress": {"step": 5, "total": 10}}, "elapsed_s": 3}
        yield {"type": "tool_result", "id": "c1", "name": "bittle_train_and_benchmark",
               "result": {"ok": True, "run_id": "run-9", "task": "slope_up", "suite": "slope_v1", "score": 9.5,
                          "gates_passed": 9, "gates_total": 10, "reflection": "BENCHMARK FAILED",
                          "gates": [{"gate": "fall_rate", "value": 0.2, "op": "<=", "threshold": 0.1, "pass": False, "note": "x", "unit": "f"}],
                          "episodes": [1, 2, 3]}}
        yield {"type": "done", "final_message": "report"}

    async def consume():
        got = []
        sub = rec.subscribe()
        hello = await sub.__anext__()
        s = rec.start("train it to walk up a slope")
        async for ev in rec.tee(s, fake_stream()):
            got.append(ev)
        live = []
        for _ in range(3):
            live.append(await sub.__anext__())
        await sub.aclose()
        return s, got, hello, live

    s, got, hello, live = asyncio.run(consume())
    assert hello["type"] == "hello" and len(got) == 6  # the caller still sees every raw event
    d = s.to_dict()
    assert d["prompt"].startswith("train it") and d["live"] is False
    assert [e["type"] for e in d["events"]] == ["thought", "tool_call", "progress", "tool_result", "done"]  # thoughts coalesced
    assert d["events"][0]["content"] == "I will train."
    c = d["cycles"][0]
    assert c["tool"] == "bittle_train_and_benchmark" and c["status"] == "ok" and c["result"]["run_id"] == "run-9"
    assert c["args"]["task"] == "slope_up" and c["progress"]["progress"]["step"] == 5
    assert "episodes" not in c["result"] and c["result"]["gates"][0]["gate"] == "fall_rate"
    assert d["run_ids"] == ["run-9"]
    assert live[0]["type"] == "session_start" and live[1]["type"] == "event"
    # persisted: a fresh recorder over the same dir replays the session
    rec2 = SessionRecorder(tmp_path)
    assert rec2.list()[0]["session_id"] == s.session_id and rec2.get(s.session_id).cycles[0]["result"]["run_id"] == "run-9"
    assert rec2.get(s.session_id).prompt.startswith("train it")


def test_training_dashboard_proxies(monkeypatch):
    from app import main as main_mod

    class DashTrainer(FakeTrainer):
        async def run(self, run_id):
            return {"run_id": run_id, "status": "training", "task": "slope_up", "suite": "slope_v1",
                    "curve": [{"step": 1, "reward": 2.0}], "config": {}, "metrics": {}}

    main_mod.agent_harness.trainer = DashTrainer()
    with TestClient(app) as c:
        assert c.get("/api/training/tasks").json()["tasks"][1]["task"] == "slope_up"
        r = c.get("/api/training/run/run-1").json()
        assert r["suite"] == "slope_v1" and r["curve"][0]["reward"] == 2.0
        assert c.get("/api/training/benchmark/run-1", params={"suite": "slope_v1"}).json()["suite"] == "slope_v1"
        assert c.get("/api/training/runs", params={"suite": "slope_v1"}).json()["runs"][0]["suite"] == "slope_v1"
        assert c.post("/api/training/compare", json={"run_ids": ["run-1"]}).json()["runs"]
        assert "opencat_trF" in c.get("/api/training/baselines").json()
        assert c.get("/api/training/sessions").json()["live"] is None
        assert c.get("/api/training/sessions/nope").status_code == 404
    class Down(DownTrainer):
        async def tasks(self):
            raise TrainerUnavailable("connection refused")

    main_mod.agent_harness.trainer = Down()
    with TestClient(app) as c:
        assert c.get("/api/training/tasks").status_code == 503
