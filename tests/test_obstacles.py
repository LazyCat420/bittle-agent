"""Tests for Obstacle Courses, Terrain Geometry, and Kinematic Clearance Evaluation."""

import pytest
import asyncio
from starlette.testclient import TestClient

from app.agent import GLMAgentHarness
from app.config import Settings
from app.controller import Controller
from app.main import app
from app.motion.obstacles import (
    COURSE_PRESETS,
    COURSE_LAYOUTS,
    get_course_layout,
    evaluate_terrain_clearance,
)
from app.motion.builtin_library import get_builtin_moveset, STAND_ANGLES


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def harness():
    settings = Settings(
        llm_api_base="http://127.0.0.1:11434/v1",
        llm_api_key="ollama",
        llm_model="glm4:latest",
        allow_real_hardware=False,
        allow_locomotion=True,
    )
    ctrl = Controller(settings=settings)
    return GLMAgentHarness(ctrl, settings)


def test_obstacle_presets_and_layouts():
    """Verify all defined course presets have valid geometry layouts."""
    for preset_name in COURSE_PRESETS:
        layout = get_course_layout(preset_name)
        assert layout["preset"] == preset_name
        assert "description" in layout

    stairs = get_course_layout("mini_stairs")
    assert stairs["riser_height_m"] == 0.018
    assert stairs["step_count"] == 3

    tunnel = get_course_layout("crawl_tunnel")
    assert tunnel["ceiling_height_m"] == 0.065

    ramp = get_course_layout("ramp_bridge")
    assert ramp["ramp_height_m"] == 0.040


def test_stair_clearance_evaluation():
    """Verify stair evaluation detects foot lift insufficiency vs success."""
    # Standard stand has knees at 80° - foot lift is 0mm, trips on 18mm riser
    stand_res = evaluate_terrain_clearance("mini_stairs", STAND_ANGLES)
    assert stand_res["clears"] is False
    assert stand_res["margin_mm"] < 0
    assert any("Insufficient foot lift" in issue for issue in stand_res["issues"])

    # High knee flexion (102°) achieves >18mm lift
    stair_moveset = get_builtin_moveset("stair_step_up")
    assert stair_moveset is not None
    stair_res = evaluate_terrain_clearance("mini_stairs", stair_moveset["frames"])
    assert stair_res["clears"] is True
    assert stair_res["margin_mm"] >= 0


def test_crawl_tunnel_clearance_evaluation():
    """Verify crawl tunnel detects standing collision vs belly crawl clearance."""
    # Standard stand has height ~88mm, which exceeds 65mm ceiling
    stand_res = evaluate_terrain_clearance("crawl_tunnel", STAND_ANGLES)
    assert stand_res["clears"] is False
    assert stand_res["margin_mm"] < 0
    assert any("exceeds tunnel ceiling" in issue for issue in stand_res["issues"])

    # Low tunnel crawl moveset keeps height under 60mm
    tunnel_moveset = get_builtin_moveset("low_tunnel_crawl")
    assert tunnel_moveset is not None
    tunnel_res = evaluate_terrain_clearance("crawl_tunnel", tunnel_moveset["frames"])
    assert tunnel_res["clears"] is True
    assert tunnel_res["margin_mm"] > 0


def test_ramp_bridge_clearance_evaluation():
    """Verify ramp evaluation checks pitch balance."""
    # Extreme backwards pitch (front knees 105, rear knees 65 -> differential 40)
    tipped_frame = {12: 105, 13: 105, 14: 65, 15: 65}
    tipped_res = evaluate_terrain_clearance("ramp_bridge", tipped_frame)
    assert tipped_res["clears"] is False

    # Ramp climb moveset
    ramp_moveset = get_builtin_moveset("ramp_climb")
    assert ramp_moveset is not None
    ramp_res = evaluate_terrain_clearance("ramp_bridge", ramp_moveset["frames"])
    assert ramp_res["clears"] is True


def test_agility_slalom_evaluation():
    """Verify slalom requires steering / yaw weaving."""
    # Straight walk without steering
    straight_frames = [
        {"angles": STAND_ANGLES},
        {"angles": STAND_ANGLES},
        {"angles": STAND_ANGLES},
    ]
    straight_res = evaluate_terrain_clearance("agility_slalom", straight_frames)
    assert straight_res["clears"] is False

    # Weave frames with head pan and differential stance
    weave_left = get_builtin_moveset("slalom_weave_left")
    assert weave_left is not None
    weave_res = evaluate_terrain_clearance("agility_slalom", weave_left["frames"])
    assert weave_res["clears"] is True


def test_harness_obstacle_tools(harness):
    """Test GLM agent tool execution for course loading and clearance."""
    # 1. Load course
    load_res = asyncio.run(harness.execute_tool("bittle_load_course", {"preset": "mini_stairs"}))
    assert load_res["ok"] is True
    assert load_res["preset"] == "mini_stairs"
    assert harness.active_course == "mini_stairs"

    # 2. Get layout
    layout_res = asyncio.run(harness.execute_tool("bittle_get_course_layout", {}))
    assert layout_res["ok"] is True
    assert layout_res["preset"] == "mini_stairs"
    assert layout_res["layout"]["step_count"] == 3

    # 3. Evaluate terrain clearance via tool using moveset_name
    eval_res = asyncio.run(harness.execute_tool("bittle_evaluate_terrain_clearance", {
        "course_preset": "mini_stairs",
        "moveset_name": "stair_step_up",
    }))
    assert eval_res["ok"] is True
    assert eval_res["clears"] is True

    # 4. Unknown course rejected
    bad_load = asyncio.run(harness.execute_tool("bittle_load_course", {"preset": "volcano_jump"}))
    assert bad_load["ok"] is False


def test_api_obstacle_endpoints(client):
    """Test REST API endpoints for obstacles."""
    # GET presets
    presets_res = client.get("/api/obstacles/presets")
    assert presets_res.status_code == 200
    data = presets_res.json()
    assert "mini_stairs" in data["presets"]
    assert "crawl_tunnel" in data["presets"]

    # GET layout
    layout_res = client.get("/api/obstacles/layout/crawl_tunnel")
    assert layout_res.status_code == 200
    ldata = layout_res.json()
    assert ldata["preset"] == "crawl_tunnel"
    assert ldata["layout"]["ceiling_height_m"] == 0.065

    # POST evaluate
    eval_res = client.post("/api/obstacles/evaluate", json={
        "course_preset": "crawl_tunnel",
        "moveset_name": "low_tunnel_crawl"
    })
    assert eval_res.status_code == 200
    edata = eval_res.json()
    assert edata["clears"] is True


def test_x_forward_obstacle_alignment():
    """Regression test ensuring obstacles are strictly positioned in front of Bittle (+X axis)."""
    for preset_name in ["mini_stairs", "ramp_bridge", "crawl_tunnel", "agility_slalom"]:
        layout = get_course_layout(preset_name)
        assert "start_x_m" in layout, f"{preset_name} must define start_x_m"
        assert layout["start_x_m"] > 0, f"{preset_name} must start forward of Bittle nose (x > 0)"

        for obs in layout["obstacles"]:
            if "min_x" in obs and "max_x" in obs:
                assert obs["min_x"] >= layout["start_x_m"]
                assert obs["max_x"] > obs["min_x"]
            elif "start_x" in obs and "end_x" in obs:
                assert obs["start_x"] >= layout["start_x_m"]
                assert obs["end_x"] > obs["start_x"]
            elif "x" in obs:
                assert obs["x"] >= layout["start_x_m"]

