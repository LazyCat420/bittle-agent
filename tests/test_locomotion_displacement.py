"""Tests for Forward/Backward Locomotion Displacement & Kinematic Verification.

Validates that forward strides advance the robot along +X, backward strides retreat
along -X, turns rotate yaw accurately, and elevation does not penetrate the ground plane.
"""

import math
import pytest
import asyncio
from app.agent import GLMAgentHarness
from app.config import Settings
from app.controller import Controller
from app.motion.builtin_library import get_builtin_moveset


@pytest.fixture
def agent_harness():
    cfg = Settings(allow_real_hardware=False, llm_api_base="http://10.0.0.141:8000/v1", llm_model="GLM-5.3-Flash-EXL3")
    ctrl = Controller(cfg)
    ctrl.safety.clear_estop()
    return GLMAgentHarness(ctrl, ctrl.settings)


def test_forward_backward_navigate_step(agent_harness):
    """Test taking forward steps advances position, and backward steps retreats position."""
    # 1. Reset pose
    reset_res = asyncio.run(agent_harness.execute_tool("bittle_reset_pose", {}))
    assert reset_res["ok"] is True
    assert reset_res["pose"]["x"] == 0.0
    assert reset_res["pose"]["z"] == 0.0
    assert reset_res["pose"]["yaw"] == 0.0

    # 2. Advance 3 steps forward
    fwd_res = asyncio.run(agent_harness.execute_tool("bittle_navigate_step", {"direction": "forward", "count": 3}))
    assert fwd_res["ok"] is True
    assert fwd_res["steps_taken"] == 3
    x_after_fwd = fwd_res["pose"]["x"]
    assert x_after_fwd > 0.10, f"Expected x > 0.10m after 3 forward steps, got {x_after_fwd}"
    assert fwd_res["pose"]["y"] >= 0.0532, "Robot elevation must stay at or above ground plane baseline"

    # 3. Retreat 2 steps backward
    back_res = asyncio.run(agent_harness.execute_tool("bittle_navigate_step", {"direction": "backward", "count": 2}))
    assert back_res["ok"] is True
    assert back_res["steps_taken"] == 2
    x_after_back = back_res["pose"]["x"]
    assert x_after_back < x_after_fwd, f"Expected position to decrease after backward step: {x_after_back} vs {x_after_fwd}"
    assert x_after_back > 0.0, "Net position should still be positive (3 fwd - 2 back)"
    assert back_res["pose"]["y"] >= 0.0532, "Robot elevation must not penetrate floor after backward strides"


def test_turn_left_and_right_yaw_changes(agent_harness):
    """Test turning left rotates yaw counter-clockwise (+deg), turning right rotates clockwise (-deg)."""
    asyncio.run(agent_harness.execute_tool("bittle_reset_pose", {}))

    # Turn left
    tl_res = asyncio.run(agent_harness.execute_tool("bittle_navigate_step", {"direction": "turn_left", "count": 2}))
    assert tl_res["ok"] is True
    assert tl_res["pose"]["yaw"] == pytest.approx(30.0, abs=1e-2)

    # Turn right 3 times (net -15 deg from 30 = -15 deg, wrapped to 345.0 in [0, 360))
    tr_res = asyncio.run(agent_harness.execute_tool("bittle_navigate_step", {"direction": "turn_right", "count": 3}))
    assert tr_res["ok"] is True
    assert tr_res["pose"]["yaw"] == pytest.approx(345.0, abs=1e-2)


def test_builtin_movesets_include_wkR_and_bk():
    """Verify both forward, backward, left, and right gaits exist in the builtin library."""
    for skill_name in ["wkF", "bk", "wkL", "wkR", "trF", "crF"]:
        m = get_builtin_moveset(skill_name)
        assert m is not None, f"Builtin moveset '{skill_name}' must exist"
        assert len(m["frames"]) >= 4, f"Moveset '{skill_name}' must have at least 4 keyframes"
        assert m["kind"] == "gait", f"Moveset '{skill_name}' must be classified as gait"


def test_viewer_kinematic_equations():
    """Verify mathematical consistency of the 3D displacement integrator used in viewer.js.

    x' = x + vx * dt * cos(yaw)
    z' = z - vx * dt * sin(yaw)
    yaw' = yaw + vyaw * dt
    """
    dt = 0.016  # 60 FPS frame time
    total_time = 1.0  # 1 second of forward walking
    steps = int(total_time / dt)

    # Test forward walking from origin
    vx_fwd = 0.075
    x, z, yaw = 0.0, 0.0, 0.0
    for _ in range(steps):
        x += vx_fwd * dt * math.cos(yaw)
        z -= vx_fwd * dt * math.sin(yaw)

    assert x == pytest.approx(0.075, abs=1e-3)
    assert z == pytest.approx(0.0, abs=1e-3)

    # Test backward walking from x = 0.075
    vx_back = -0.055
    for _ in range(steps):
        x += vx_back * dt * math.cos(yaw)
        z -= vx_back * dt * math.sin(yaw)

    assert x == pytest.approx(0.020, abs=1e-3)
    assert z == pytest.approx(0.0, abs=1e-3)

    # Test walking forward while turning left (+90 degrees)
    x, z, yaw = 0.0, 0.0, math.pi / 2  # yaw = +90°
    for _ in range(steps):
        x += vx_fwd * dt * math.cos(yaw)
        z -= vx_fwd * dt * math.sin(yaw)

    assert x == pytest.approx(0.0, abs=1e-3)
    assert z == pytest.approx(-0.075, abs=1e-3)  # In Three.js, +90° yaw points toward -Z
