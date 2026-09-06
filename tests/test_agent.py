"""Tests for GLM Agent Harness."""

import asyncio
import pytest
from fastapi.testclient import TestClient

from app.agent import GLMAgentHarness
from app.config import Settings
from app.controller import Controller
from app.main import app


@pytest.fixture
def sim_controller():
    cfg = Settings(allow_real_hardware=False)
    ctrl = Controller(cfg)
    ctrl.safety.clear_estop()
    return ctrl


@pytest.fixture
def harness(sim_controller):
    return GLMAgentHarness(sim_controller, sim_controller.settings)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_agent_list_capabilities(harness):
    caps = asyncio.run(harness.execute_tool("bittle_list_capabilities", {}))
    assert caps["ok"] is True
    assert len(caps["skills"]) > 0
    # Standard factory Bittle has 9 controllable servos (head pan 0 + 8 leg joints)
    assert len(caps["joints"]) == 9


def test_agent_hardware_profile_tools(harness):
    prof = asyncio.run(harness.execute_tool("bittle_get_hardware_profile", {}))
    assert prof["ok"] is True
    assert prof["robot_model"] == "BITTLE"
    assert 0 in prof["installed_joints"]
    assert 1 not in prof["installed_joints"]

    envs = asyncio.run(harness.execute_tool("bittle_get_joint_envelopes", {}))
    assert envs["ok"] is True
    assert 0 in envs["envelopes"]
    assert envs["envelopes"][0]["agent"] == [-60, 60]


def test_agent_skill_authoring_lifecycle_tools(harness):
    # 1. Draft
    draft_res = asyncio.run(harness.execute_tool("bittle_draft_skill", {
        "skill": {
            "name": "nod_head_safe",
            "frames": [
                {"angles_deg": {0: 20}, "speed_deg_per_step": 4, "delay_ms": 100},
                {"angles_deg": {0: -20}, "speed_deg_per_step": 4, "delay_ms": 100},
                {"angles_deg": {0: 0}, "speed_deg_per_step": 4, "delay_ms": 100},
            ]
        }
    }))
    assert draft_res["ok"] is True
    ir_data = draft_res["skill_ir"]

    # 2. Validate
    val_res = asyncio.run(harness.execute_tool("bittle_validate_skill", {"skill": ir_data}))
    assert val_res["ok"] is True
    m_hash = val_res["manifest_hash"]
    assert m_hash is not None

    # 3. Simulate
    sim_res = asyncio.run(harness.execute_tool("bittle_simulate_skill", {"manifest_hash": m_hash}))
    assert sim_res["ok"] is True
    assert sim_res["status"] == "simulated"
    assert sim_res["frames"] == 3


def test_agent_status(harness):
    res = asyncio.run(harness.execute_tool("bittle_status", {"target": "sim"}))
    assert res["ok"] is True
    assert "angles" in res
    assert "status" in res
    assert res["status"]["target"]["backend"] == "sim"


def test_agent_do_skill_posture(harness):
    res = asyncio.run(harness.execute_tool("bittle_do_skill", {"skill": "sit", "target": "sim"}))
    assert res["ok"] is True
    assert res["skill"] == "sit"
    assert res["token"] == "ksit"


def test_agent_do_skill_gait_requires_ack(harness):
    # Locomoting gait without ack must be refused by safety layer
    res = asyncio.run(harness.execute_tool("bittle_do_skill", {"skill": "wkF", "target": "sim", "ack_locomotion": False}))
    assert res["ok"] is False
    assert res["error"] == "safety_refused"
    assert res["reason"] == "locomotion_unacked"


def test_agent_do_skill_gait_with_ack(harness):
    res = asyncio.run(harness.execute_tool("bittle_do_skill", {"skill": "wkF", "target": "sim", "ack_locomotion": True}))
    assert res["ok"] is True
    assert res["skill"] == "wkF"


def test_agent_move_joints_with_clamp(harness):
    # Requesting joint 0 (head) at 150 deg -> clamped to 120
    res = asyncio.run(harness.execute_tool("bittle_move_joints", {"angles": {"head": 150}, "target": "sim"}))
    assert res["ok"] is True
    assert res["clamped"] is True
    assert len(res["adjustments"]) == 1
    assert res["adjustments"][0]["applied"] == 120
    assert res["applied"]["0"] == 120


def test_agent_estop(harness):
    res = asyncio.run(harness.execute_tool("bittle_estop", {"reason": "test_agent_stop"}))
    assert res["ok"] is True
    assert res["estop"] == "engaged"
    # Further motions must be rejected due to E-Stop
    move_res = asyncio.run(harness.execute_tool("bittle_do_skill", {"skill": "sit", "target": "sim"}))
    assert move_res["ok"] is False
    assert move_res["reason"] == "estop_engaged"
    # Clear estop
    harness.controller.clear_estop()


def test_api_agent_config(client):
    res = client.get("/api/agent/config")
    assert res.status_code == 200
    data = res.json()
    assert "api_base" in data
    assert "model" in data


def test_api_agent_step(client):
    res = client.post("/api/agent/step", json={"tool": "bittle_do_skill", "args": {"skill": "balance"}})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["skill"] == "balance"


def test_resolve_endpoint_and_model_default(harness):
    base, model = asyncio.run(harness.resolve_endpoint_and_model())
    assert base.startswith("http")
    assert model != ""


def test_resolve_endpoint_auto_discovery_mock(harness, monkeypatch):
    import httpx

    class MockResponse:
        status_code = 200
        def json(self):
            return {"data": [{"id": "GLM-5.3-Flash-EXL3"}]}

    async def mock_get(self, url, *args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    base, model = asyncio.run(harness.resolve_endpoint_and_model())
    assert model == "GLM-5.3-Flash-EXL3"


def test_agent_chat_stream_error_schema_conformance(harness, monkeypatch):
    import httpx

    async def mock_resolve():
        return "http://test-cluster:8000/v1", "GLM-5.3-Flash-EXL3"

    monkeypatch.setattr(harness, "resolve_endpoint_and_model", mock_resolve)

    def mock_stream(self, method, url, **kwargs):
        raise httpx.ConnectError("Connection refused to test cluster")

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    async def collect_events():
        events = []
        async for ev in harness.chat_stream([{"role": "user", "content": "hello"}]):
            events.append(ev)
        return events

    events = asyncio.run(collect_events())
    assert len(events) >= 1
    err_ev = events[0]
    assert err_ev["type"] == "error"
    # Must contain error, detail, and content so UI never renders undefined
    assert "error" in err_ev and err_ev["error"]
    assert "detail" in err_ev and err_ev["detail"]
    assert "content" in err_ev and err_ev["content"]


def test_agent_chat_stream_streaming_reasoning_and_text(harness, monkeypatch):
    import httpx
    import json

    async def mock_resolve():
        return "http://test-cluster:8000/v1", "GLM-5.3-Flash-EXL3"

    monkeypatch.setattr(harness, "resolve_endpoint_and_model", mock_resolve)

    sse_lines = [
        'data: ' + json.dumps({
            "choices": [{"delta": {"reasoning": "Analyzing robot stability..."}}]
        }),
        'data: ' + json.dumps({
            "choices": [{"delta": {"content": "I am standing steady."}}]
        }),
        'data: [DONE]'
    ]

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def raise_for_status(self):
            pass
        async def aiter_lines(self):
            for line in sse_lines:
                yield line

    def mock_stream(self, method, url, **kwargs):
        return MockStreamResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    async def collect_events():
        events = []
        async for ev in harness.chat_stream([{"role": "user", "content": "status"}]):
            events.append(ev)
        return events

    events = asyncio.run(collect_events())
    types = [e["type"] for e in events]
    assert "thought" in types
    assert "text" in types
    thought_ev = next(e for e in events if e["type"] == "thought")
    assert "Analyzing robot stability..." in thought_ev["content"]
    text_ev = next(e for e in events if e["type"] == "text")
    assert "I am standing steady." in text_ev["content"]


def test_agent_chat_stream_streaming_tool_call(harness, monkeypatch):
    import httpx
    import json

    async def mock_resolve():
        return "http://test-cluster:8000/v1", "GLM-5.3-Flash-EXL3"

    monkeypatch.setattr(harness, "resolve_endpoint_and_model", mock_resolve)

    # Turn 1: tool call -> Turn 2: explanation
    turn1_lines = [
        'data: ' + json.dumps({
            "choices": [{"delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_balance_1",
                    "function": {"name": "bittle_do_skill", "arguments": "{\"skill\": \"balance\"}"}
                }]
            }}]
        }),
        'data: [DONE]'
    ]

    turn2_lines = [
        'data: ' + json.dumps({
            "choices": [{"delta": {"content": "Robot is now balanced."}}]
        }),
        'data: [DONE]'
    ]

    calls = 0

    class MockStreamTurnResponse:
        def __init__(self, lines):
            self.status_code = 200
            self.lines = lines
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            for line in self.lines:
                yield line

    def mock_stream(self, method, url, **kwargs):
        nonlocal calls
        calls += 1
        lines = turn1_lines if calls == 1 else turn2_lines
        return MockStreamTurnResponse(lines)

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    async def collect_events():
        events = []
        async for ev in harness.chat_stream([{"role": "user", "content": "please balance"}]):
            events.append(ev)
        return events

    events = asyncio.run(collect_events())
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "text" in types
    assert "done" in types

    tc = next(e for e in events if e["type"] == "tool_call")
    assert tc["name"] == "bittle_do_skill"
    assert tc["args"] == {"skill": "balance"}

    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["name"] == "bittle_do_skill"
    assert tr["result"]["ok"] is True


