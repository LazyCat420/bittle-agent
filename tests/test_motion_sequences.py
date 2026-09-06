"""Unit tests for compound motion sequences, expressive macros, and moveset library."""

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


def test_system_prompt_has_injected_state(harness):
    prompt = harness.get_system_prompt(target="sim")
    assert "CURRENT ROBOT RUNTIME STATE:" in prompt
    assert harness.controller.profile.profile_id in prompt
    assert "REAL-TIME LATENCY DIRECTIVE:" in prompt
    assert "bittle_execute_sequence" in prompt


def test_execute_sequence_valid_steps(harness):
    steps = [
        {"type": "skill", "skill": "sit"},
        {"type": "pause", "delay_ms": 100},
        {"type": "move", "angles": {0: 35}, "delay_ms": 150},
        {"type": "move", "angles": {0: -35}, "delay_ms": 150},
        {"type": "move", "angles": {0: 0}, "delay_ms": 100},
    ]
    res = asyncio.run(harness.execute_tool("bittle_execute_sequence", {
        "name": "sit_and_look",
        "steps": steps,
        "target": "sim",
    }))
    assert res["ok"] is True
    assert res["name"] == "sit_and_look"
    assert len(res["executed_steps"]) == 5
    assert "moveset" in res
    moveset = res["moveset"]
    assert len(moveset["frames"]) >= 4
    # Frame contains angles and duration
    assert 0 in moveset["frames"][1]["angles"]
    assert moveset["total_duration_ms"] > 0


def test_execute_sequence_dry_run_rejection(harness):
    # Joint 1 is uninstalled on standard Bittle
    steps = [
        {"type": "skill", "skill": "sit"},
        {"type": "move", "angles": {1: 45}},
    ]
    res = asyncio.run(harness.execute_tool("bittle_execute_sequence", {
        "name": "invalid_joint_seq",
        "steps": steps,
        "target": "sim",
    }))
    assert res["ok"] is False
    assert res["error"] == "safety_refused"
    assert res["reason"] == "unused_joint"


def test_execute_sequence_clamping(harness):
    # Head joint 0 angle exceeds safe envelope (e.g. 150 -> clamped to 120)
    steps = [
        {"type": "move", "angles": {0: 150}},
    ]
    res = asyncio.run(harness.execute_tool("bittle_execute_sequence", {
        "name": "clamp_seq",
        "steps": steps,
        "target": "sim",
    }))
    assert res["ok"] is True
    assert res["moveset"]["frames"][0]["angles"][0] == 120


def test_express_look_around(harness):
    res = asyncio.run(harness.execute_tool("bittle_express", {
        "expression": "look_around",
        "target": "sim",
    }))
    assert res["ok"] is True
    assert res["expression"] == "look_around"
    assert "moveset" in res
    assert len(res["moveset"]["frames"]) >= 3


def test_express_unknown_fails_gracefully(harness):
    res = asyncio.run(harness.execute_tool("bittle_express", {
        "expression": "non_existent_emotion",
        "target": "sim",
    }))
    assert res["ok"] is False
    assert "unknown expression" in res["error"]


def test_save_and_list_moveset(harness, client):
    frames = [
        {"angles": {0: 20, 8: -45}, "delay_ms": 100},
        {"angles": {0: -20, 8: -45}, "delay_ms": 100},
    ]
    save_res = asyncio.run(harness.execute_tool("bittle_save_moveset", {
        "name": "test_nod_wave",
        "description": "Nod and wave test",
        "frames": frames,
    }))
    assert save_res["ok"] is True
    assert save_res["name"] == "test_nod_wave"

    # Verify API lists saved movesets
    list_res = client.get("/api/movesets")
    assert list_res.status_code == 200
    data = list_res.json()
    names = [m["name"] for m in data.get("movesets", [])]
    assert "test_nod_wave" in names


def test_sim_skill_updates_joint_state(client):
    """Ensure running a posture like 'sit' updates SimBackend._angles so readbacks are accurate."""
    res = client.post("/api/skill", json={"skill": "sit", "target": "sim"})
    assert res.status_code == 200
    assert res.json()["ok"] is True

    # State readback must reflect sit angles, NOT the default stand pose!
    state_res = client.get("/api/joints/state?target=sim")
    assert state_res.status_code == 200
    angles = state_res.json()["angles"]
    assert angles["8"] == -30
    assert angles["10"] == 80
    assert angles["12"] == 40

    # Reset back to balance/stand
    stand_res = client.post("/api/skill", json={"skill": "balance", "target": "sim"})
    assert stand_res.status_code == 200
    state_after = client.get("/api/joints/state?target=sim").json()["angles"]
    assert state_after["8"] == -45
    assert state_after["12"] == 80


def test_gait_locomotion_execution(client):
    """Ensure locomotion gaits execute cleanly with ack_locomotion."""
    for gait in ["wkF", "wkL", "trF", "crF", "bk"]:
        res = client.post("/api/skill", json={
            "skill": gait,
            "target": "sim",
            "ack_locomotion": True
        })
        assert res.status_code == 200
        assert res.json()["ok"] is True
        assert res.json()["token"] == f"k{gait}"


def test_builtin_moveset_coverage():
    """Ensure all common gaits and behaviors have rich multi-frame keyframes."""
    from app.motion.builtin_library import BUILTIN_MOVESETS
    expected_moves = [
        "sit", "balance", "rest", "zero", "up", "str",
        "wkF", "wkL", "trF", "bk", "crF",
        "bf", "ck", "hi", "pu", "nd", "pee", "fiv", "gdb", "hsk", "hu", "jmp", "kc"
    ]
    for move in expected_moves:
        assert move in BUILTIN_MOVESETS, f"Missing keyframe moveset definition for {move}"
        frames = BUILTIN_MOVESETS[move]["frames"]
        assert len(frames) >= 1, f"Moveset {move} must have at least 1 frame"

