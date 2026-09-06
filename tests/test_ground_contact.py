"""Tests for Dynamic Gravity, Posture Ground Contact, and Sitting Kinematics."""

import pytest
from app.motion.obstacles import compute_leg_reach, compute_posture_ground_contact
from app.motion.builtin_library import STAND_ANGLES, SIT_ANGLES, REST_ANGLES


def test_stand_pose_ground_contact():
    """Standing pose must ground all four feet with level chassis ~53.2mm high."""
    contact = compute_posture_ground_contact(STAND_ANGLES)
    assert contact["is_standing"] is True
    assert contact["is_sitting"] is False
    assert contact["is_resting"] is False
    assert 52.0 <= contact["chassis_height_mm"] <= 55.0
    assert abs(contact["pitch_deg"]) < 1.0


def test_sit_pose_ground_contact():
    """Sitting pose must drop rear rump to ground (22mm), lower chassis height, and engage sitting flag."""
    contact = compute_posture_ground_contact(SIT_ANGLES)
    assert contact["is_sitting"] is True
    assert contact["is_standing"] is False
    assert contact["is_resting"] is False

    # Rear rump rests on floor at ~22mm contact floor
    assert contact["h_rear_m"] == pytest.approx(0.0220, abs=0.002)
    # Overall chassis height is lowered from 53.2mm to ~25.2mm under gravity
    assert contact["chassis_height_mm"] < 35.0


def test_rest_pose_ground_contact():
    """Resting pose (relax on belly) drops entire body flat to floor level."""
    contact = compute_posture_ground_contact(REST_ANGLES)
    assert contact["is_resting"] is True
    assert contact["is_sitting"] is False
    assert contact["is_standing"] is False

    # Belly resting on floor
    assert contact["chassis_height_mm"] <= 32.0


def test_crouch_pose_ground_contact():
    """Crouch pose drops front paws to ~28mm, significantly lower than standing."""
    crouch_angles = {**STAND_ANGLES, 8: -30, 9: -30, 12: 40, 13: 40}
    contact = compute_posture_ground_contact(crouch_angles)
    assert contact["h_front_m"] < 0.035
    assert contact["h_front_m"] < contact["h_rear_m"]
