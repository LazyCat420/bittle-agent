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

    async def submit(self, patch, name="", base_run_id=None, notes=""):
        self.calls.append(("submit", patch, base_run_id))
        return {"run_id": "run-1", "status": "queued", "config_hash": "h1", "diff": []}

    async def status(self, run_id, wait_s=0.0, until="any"):
        self.calls.append(("status", run_id, wait_s, until))
        st = self.status_seq[min(self.n, len(self.status_seq) - 1)]
        self.n += 1
        return {"run_id": run_id, "status": st, "progress": {"step": 10, "total": 100}, "metrics": {"reward_final": 5.0},
                "curve": [{"step": 10, "reward": 5.0}]}

    async def benchmark(self, run_id, suite="flat_v1", n_episodes=None, dr_sweep=False, dual_sim=False, wait_s=0.0):
        self.calls.append(("benchmark", run_id, dr_sweep))
        return {"run_id": run_id, "status": "benchmarking", "queued": True}

    async def get_benchmark(self, run_id, suite="flat_v1"):
        return {"run_id": run_id, "suite": suite, "suite_version": "1.0.0", "gates_passed": 11, "gates_total": 12,
                "passed": False, "score": 11.3, "reflection": "BENCHMARK FAILED 11/12 gates.",
                "gates": [{"gate": "fall_rate", "value": 0.2, "op": "<=", "threshold": 0.1, "pass": False, "note": "hint", "unit": "x"}],
                "metrics": {"fall_rate": 0.2, "forward_distance_p50": 0.7}, "episodes": [{"seed": 0}]}

    async def list_runs(self, sort="score", limit=20):
        return {"runs": [{"run_id": "run-1", "score": 11.3}], "baselines": {"opencat_trF": {"score": 5.0}}, "best": {"run_id": "run-1"}}

    async def compare(self, run_ids):
        return {"runs": {r: {} for r in run_ids}, "gates_table": {}, "config_diffs_vs_first": {}}

    async def baselines(self):
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
