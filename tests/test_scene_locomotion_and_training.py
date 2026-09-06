"""Tests for Bittle 3D Scene Locomotion, Step Navigation, and Stair Training Runs."""

import pytest
import asyncio
from app.agent import GLMAgentHarness
from app.config import Settings
from app.controller import Controller
from app.backends.sim import SimBackend
from app.motion.obstacles import simulate_stair_climb_episode, get_obstacle_proximity
from app.motion.builtin_library import STAND_ANGLES, get_builtin_moveset


@pytest.fixture
def agent_harness():
    cfg = Settings(allow_real_hardware=False, llm_api_base="http://10.0.0.141:8000/v1", llm_model="GLM-5.3-Flash-EXL3")
    ctrl = Controller(cfg)
    ctrl.safety.clear_estop()
    return GLMAgentHarness(ctrl, ctrl.settings)


def test_obstacle_proximity_calculation():
    """Verify obstacle proximity accurately identifies stairs and steps along +X."""
    # Approach zone
    p_approach = get_obstacle_proximity("mini_stairs", current_x=0.05)
    assert p_approach["current_zone"] == "approach"
    assert p_approach["next_obstacle"] == "step_1"
    assert p_approach["distance_to_riser_m"] == pytest.approx(0.090, rel=1e-2)
    assert p_approach["current_elevation_mm"] == 0.0

    # Step 1 zone
    p_step1 = get_obstacle_proximity("mini_stairs", current_x=0.180)
    assert p_step1["current_zone"] == "step_1"
    assert p_step1["next_obstacle"] == "step_2"
    assert p_step1["current_elevation_mm"] == 18.0

    # Top landing platform
    p_top = get_obstacle_proximity("mini_stairs", current_x=0.420)
    assert p_top["current_zone"] == "top_landing_platform"
    assert p_top["current_elevation_mm"] == 54.0


def test_simulate_stair_climb_insufficient_lift():
    """Standard stand or low knee flexion (<18mm lift) trips on Step 1."""
    # Neutral stand has knee at 80° -> lift is 0mm
    res = simulate_stair_climb_episode(STAND_ANGLES)
    assert res["success"] is False
    assert res["outcome"] == "tripped"
    assert res["failed_at_step"] == 1
    assert res["deficit_mm"] >= 15.0
    assert "EPISODE FAILED" in res["reflection"]
    assert "guidance" in res


def test_simulate_stair_climb_success():
    """stair_step_up with knee flexion 102° clears all 3 steps onto landing platform."""
    stair_moveset = get_builtin_moveset("stair_step_up")
    assert stair_moveset is not None
    res = simulate_stair_climb_episode(stair_moveset["frames"])
    assert res["success"] is True
    assert res["outcome"] == "climbed_stairs_successfully"
    assert res["steps_cleared"] == 3
    assert res["final_elevation_mm"] == 54.0
    assert res["stability_score"] > 70.0
    assert "EPISODE SUCCESS" in res["reflection"]


def test_simulate_stair_climb_with_adjustments():
    """Adjustments dynamically boost foot lift from failing to passing."""
    # A marginal gait with knee at 88° -> lift ~6.8mm, trips
    marginal_frame = {**STAND_ANGLES, 12: 88, 13: 88}
    fail_res = simulate_stair_climb_episode(marginal_frame)
    assert fail_res["success"] is False

    # Apply +15° knee lift adjustment -> knee 103° -> lift > 18mm, clears
    pass_res = simulate_stair_climb_episode(marginal_frame, adjustments={"knee_lift_deg": 15.0})
    assert pass_res["success"] is True
    assert pass_res["steps_cleared"] == 3


def test_agent_scene_pose_and_navigation(agent_harness):
    """Test GLM agent tools: get_scene_pose, navigate_step, and reset_pose."""
    # 1. Initial pose at origin
    pose_res = asyncio.run(agent_harness.execute_tool("bittle_get_scene_pose", {}))
    assert pose_res["ok"] is True
    assert pose_res["pose"]["x"] == 0.0
    assert pose_res["pose"]["z"] == 0.0

    # 2. Navigate forward 2 steps
    nav_res = asyncio.run(agent_harness.execute_tool("bittle_navigate_step", {"direction": "forward", "count": 2}))
    assert nav_res["ok"] is True
    assert nav_res["steps_taken"] == 2
    assert nav_res["pose"]["x"] > 0.06  # 2 * 0.038 = ~0.076m

    # 3. Load mini_stairs course & reset
    load_res = asyncio.run(agent_harness.execute_tool("bittle_load_course", {"preset": "mini_stairs"}))
    assert load_res["ok"] is True
    assert load_res["pose"]["x"] == 0.0
    assert agent_harness.active_course == "mini_stairs"

    # 4. Navigate towards stairs
    nav_stairs = asyncio.run(agent_harness.execute_tool("bittle_navigate_step", {"direction": "forward", "count": 4}))
    assert nav_stairs["pose"]["x"] > 0.140  # Entered stairs
    assert nav_stairs["proximity"]["current_elevation_mm"] >= 18.0

    # 5. Reset pose
    reset_res = asyncio.run(agent_harness.execute_tool("bittle_reset_pose", {}))
    assert reset_res["ok"] is True
    assert reset_res["pose"]["x"] == 0.0


def test_agent_run_stair_training_episode(agent_harness):
    """Test GLM tool: bittle_run_stair_training_episode."""
    # Run episode with stair_step_up
    train_res = asyncio.run(agent_harness.execute_tool(
        "bittle_run_stair_training_episode",
        {"moveset_name": "stair_step_up", "knee_lift_deg": 5.0}
    ))
    assert train_res["ok"] is True
    assert train_res["episode"]["success"] is True
    assert train_res["pose"]["x"] >= 0.380
    assert "reflection" in train_res
    assert "EPISODE SUCCESS" in train_res["reflection"]
