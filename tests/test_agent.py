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
    assert len(caps["joints"]) == 10  # 10 controllable joints on Bittle


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
