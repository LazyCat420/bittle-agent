import pytest

from app.motion.primitives import (
    JOINT_GROUPS,
    CompositionError,
    make_primitive,
    compose_sequential,
    compose_parallel,
    compose_blend,
    compose,
    evaluate_frames,
)
from app.motion.builtin_library import (
    BUILTIN_PRIMITIVES,
    get_builtin_primitive,
    list_builtin_primitives,
    list_builtin_primitives_by_group,
)
from app.joints import STAND_POSE


def test_joint_groups_completeness():
    assert "head" in JOINT_GROUPS
    assert "front_left" in JOINT_GROUPS
    assert "front_right" in JOINT_GROUPS
    assert "rear_left" in JOINT_GROUPS
    assert "rear_right" in JOINT_GROUPS
    assert "torso" in JOINT_GROUPS
    assert "all" in JOINT_GROUPS

    # Verify head joint is 0
    assert JOINT_GROUPS["head"] == [0]


def test_make_primitive_valid():
    prim = make_primitive(
        name="test_head_turn",
        group="head",
        frames=[
            {"angles": {0: 30}, "delay_ms": 200, "speed_deg_per_step": 6},
            {"angles": {0: -30}, "delay_ms": 200, "speed_deg_per_step": 6},
        ],
        description="Head turn test",
        tags=["head", "turn"],
    )
    assert prim["name"] == "test_head_turn"
    assert prim["group"] == "head"
    assert len(prim["frames"]) == 2
    assert prim["frames"][0]["angles"][0] == 30


def test_make_primitive_invalid_joint():
    with pytest.raises(CompositionError) as exc_info:
        make_primitive(
            name="bad_head",
            group="head",
            frames=[{"angles": {8: 45}}],  # Joint 8 is shoulder, not head
        )
    assert "joint 8 is not in group 'head'" in str(exc_info.value)


def test_compose_sequential():
    p1 = make_primitive("p1", "head", [{"angles": {0: 20}, "delay_ms": 100}])
    p2 = make_primitive("p2", "head", [{"angles": {0: -20}, "delay_ms": 100}])
    frames = compose_sequential([p1, p2], transition_blend_ms=0)
    assert len(frames) == 2
    assert frames[0]["angles"][0] == 20
    assert frames[1]["angles"][0] == -20

    # With transition blend
    frames_blend = compose_sequential([p1, p2], transition_blend_ms=150)
    assert len(frames_blend) == 3  # p1, blend frame, p2
    assert frames_blend[1]["delay_ms"] == 150
    assert frames_blend[1]["angles"][0] == 0  # (20 + (-20)) / 2 = 0


def test_compose_parallel_disjoint():
    p_head = make_primitive("p_head", "head", [
        {"angles": {0: 35}, "delay_ms": 150},
        {"angles": {0: -35}, "delay_ms": 150},
    ])
    p_fl = make_primitive("p_fl", "front_left", [
        {"angles": {8: -40, 12: 50}, "delay_ms": 150},
        {"angles": {8: -10, 12: 20}, "delay_ms": 150},
    ])
    frames = compose_parallel([p_head, p_fl])
    assert len(frames) == 2
    # Both joints are populated in each frame
    assert frames[0]["angles"][0] == 35
    assert frames[0]["angles"][8] == -40
    assert frames[0]["angles"][12] == 50


def test_compose_parallel_collision_rejection():
    p1 = make_primitive("p1", "front_left", [{"angles": {8: 20, 12: 30}}])
    p2 = make_primitive("p2", "torso", [{"angles": {8: -10, 9: 10, 10: 10, 11: -10}}])

    # Joint 8 overlaps between front_left and torso
    with pytest.raises(CompositionError) as exc_info:
        compose_parallel([p1, p2])
    assert "both touch joints [8]" in str(exc_info.value)


def test_compose_blend():
    p1 = make_primitive("p1", "head", [{"angles": {0: 30}, "delay_ms": 100}])
    p2 = make_primitive("p2", "head", [{"angles": {0: -30}, "delay_ms": 100}])
    frames = compose_blend([p1, p2], blend_frames=2, blend_delay_ms=80)
    # 1 frame p1 + 2 blend frames + 1 frame p2 = 4 frames
    assert len(frames) == 4
    assert frames[0]["angles"][0] == 30
    assert frames[3]["angles"][0] == -30
    assert frames[1]["delay_ms"] == 80


def test_evaluate_frames():
    frames = [
        {"angles": {0: 0, 8: 0}, "delay_ms": 100},
        {"angles": {0: 20, 8: 10}, "delay_ms": 100},
        {"angles": {0: 40, 8: 20}, "delay_ms": 100},
    ]
    metrics = evaluate_frames(frames)
    assert metrics["frame_count"] == 3
    assert metrics["total_duration_ms"] == 300
    assert metrics["smoothness"] > 0
    assert metrics["max_delta_deg"] == 20.0
    assert metrics["reversal_count"] == 0


def test_builtin_primitives_validity():
    assert len(BUILTIN_PRIMITIVES) >= 15
    for name, prim in BUILTIN_PRIMITIVES.items():
        assert prim["name"] == name
        assert prim["group"] in JOINT_GROUPS
        assert len(prim["frames"]) >= 1
        for frame in prim["frames"]:
            assert "angles" in frame
            assert "delay_ms" in frame
            assert "speed_deg_per_step" in frame


def test_builtin_primitives_lookup():
    head_left = get_builtin_primitive("head_scan_left")
    assert head_left is not None
    assert head_left["group"] == "head"

    head_prims = list_builtin_primitives_by_group("head")
    assert len(head_prims) >= 2
    assert all(p["group"] == "head" for p in head_prims)
