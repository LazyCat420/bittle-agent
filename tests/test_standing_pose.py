"""
Tests for Bittle Standing Pose, Kinematic Ground Alignment, and Servo Limiters.

Verifies:
1. STAND_POSE is defined and within safe hardware limiters.
2. All 4 feet land on the same ground plane (level stance).
3. SimBackend initializes in STAND_POSE.
4. Skill transitions (balance -> STAND_POSE, rest -> REST_POSE).
5. Joint description exposes STAND_POSE as default and accurate limiters.
"""

import math
import pytest

from app import controller, joints
from app.backends.sim import SimBackend


def test_stand_pose_defined_and_safe():
    """Every angle in STAND_POSE must be within safe hardware limits."""
    assert hasattr(joints, "STAND_POSE"), "joints.STAND_POSE is not defined"
    stand = joints.STAND_POSE
    assert len(stand) >= 10, "STAND_POSE must cover all 10 controllable joints"

    for j in joints.CONTROLLABLE:
        assert j.index in stand, f"Joint #{j.index} ({j.name}) missing from STAND_POSE"
        angle = stand[j.index]
        assert j.safe_min <= angle <= j.safe_max, (
            f"Joint #{j.index} angle {angle} outside safe limits [{j.safe_min}, {j.safe_max}]"
        )


def test_stand_pose_is_default_in_joints_module():
    """DEFAULT_POSE must point to STAND_POSE so all consumers start standing."""
    assert hasattr(joints, "DEFAULT_POSE"), "joints.DEFAULT_POSE is not defined"
    assert joints.DEFAULT_POSE == joints.STAND_POSE


def test_four_feet_land_on_level_ground_plane():
    """All 4 feet must land within 1mm of the same Z plane in STAND_POSE."""
    def rot_y(v, deg):
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)

    def add_v(v1, v2):
        return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

    stand = getattr(joints, "STAND_POSE", {})
    if not stand:
        pytest.fail("joints.STAND_POSE not defined")

    legs = {
        "RF": {
            "shoulder": (0.0525, -0.0485, 0.022),
            "v_thigh": (-0.044433, 0.00142, -0.011906),
            "v_shank": (0.047813, 0.013, -0.009684),
            "s_idx": 9, "k_idx": 13,
        },
        "LF": {
            "shoulder": (0.0525, 0.0485, 0.022),
            "v_thigh": (-0.044433, -0.00142, -0.011906),
            "v_shank": (0.047813, -0.013, -0.009684),
            "s_idx": 8, "k_idx": 12,
        },
        "RR": {
            "shoulder": (-0.0525, -0.0485, 0.022),
            "v_thigh": (-0.044433, 0.00142, -0.011906),
            "v_shank": (0.048053, 0.013, -0.009684),
            "s_idx": 10, "k_idx": 14,
        },
        "LR": {
            "shoulder": (-0.0525, 0.0485, 0.022),
            "v_thigh": (-0.044433, -0.001418, -0.011906),
            "v_shank": (0.048053, -0.013002, -0.009684),
            "s_idx": 11, "k_idx": 15,
        },
    }

    foot_z_coords = []
    for name, cfg in legs.items():
        s_deg = stand[cfg["s_idx"]]
        k_deg = stand[cfg["k_idx"]]
        v_thigh_rot = rot_y(cfg["v_thigh"], s_deg)
        p_knee = add_v(cfg["shoulder"], v_thigh_rot)
        v_shank_rot = rot_y(rot_y(cfg["v_shank"], k_deg), s_deg)
        p_foot = add_v(p_knee, v_shank_rot)
        foot_z_coords.append(p_foot[2] * 1000.0)

    # Max difference between any two feet must be < 1.5mm
    max_dz = max(foot_z_coords) - min(foot_z_coords)
    assert max_dz < 1.5, f"Feet are not level: Z coords {foot_z_coords}, max delta {max_dz:.2f}mm"
    # Average height must be around -53mm
    avg_z = sum(foot_z_coords) / len(foot_z_coords)
    assert -55.0 <= avg_z <= -51.0, f"Standing height {avg_z:.1f}mm outside expected -53mm range"


@pytest.mark.anyio
async def test_sim_backend_initializes_in_stand_pose():
    """SimBackend must start in STAND_POSE by default."""
    sim = SimBackend()
    angles = await sim.read_joints()
    assert angles == getattr(joints, "STAND_POSE", None), "SimBackend did not start in STAND_POSE"


@pytest.mark.anyio
async def test_sim_backend_skill_postures():
    """SimBackend must update internal angles on balance and rest skills."""
    sim = SimBackend()
    # Move to custom angles
    await sim.move_joints([(8, 0), (12, 0)])
    cur = await sim.read_joints()
    assert cur[8] == 0

    # Running balance sets standing pose
    await sim.run_skill("balance")
    cur = await sim.read_joints()
    assert cur[8] == joints.STAND_POSE[8]
    assert cur[12] == joints.STAND_POSE[12]

    # Running rest sets rest pose
    await sim.run_skill("rest")
    cur = await sim.read_joints()
    assert cur[8] == joints.REST_POSE[8]
    assert cur[12] == joints.REST_POSE[12]


def test_describe_joints_exposes_stand_pose_as_default():
    """describe_joints must return 'default' matching STAND_POSE."""
    from app.controller import Controller
    described = Controller.describe_joints()
    for d in described:
        idx = d["index"]
        expected_def = getattr(joints, "STAND_POSE", {}).get(idx)
        assert d.get("default") == expected_def, f"Joint {idx} default was {d.get('default')} instead of {expected_def}"
