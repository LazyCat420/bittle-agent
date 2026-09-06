"""Tests for Trajectory and Motion Safety Invariants Validator."""

from __future__ import annotations

import pytest

from app.motion.schema import SkillFrame, SkillIR, SkillKind
from app.motion.validator import TrajectoryValidator
from app.profiles import get_registry


@pytest.fixture
def validator():
    return TrajectoryValidator()


def test_valid_safe_skill_passes(validator):
    ir = SkillIR(
        name="gentle_nod",
        profile_id="bittle-standard-biboard-v1-p1s",
        kind=SkillKind.BEHAVIOR,
        frames=[
            SkillFrame(angles_deg={0: 15}, speed_deg_per_step=4, delay_ms=100),
            SkillFrame(angles_deg={0: -15}, speed_deg_per_step=4, delay_ms=100),
            SkillFrame(angles_deg={0: 0}, speed_deg_per_step=4, delay_ms=100),
        ],
    )
    res = validator.validate(ir)
    assert res.valid is True
    assert len(res.errors) == 0
    assert res.budget is not None
    assert res.budget.is_bounded is True
    assert res.budget.max_delta_deg == 30.0


def test_reject_uninstalled_joint_on_standard_profile(validator):
    # Joint 1 (tail) is not on standard profile
    ir = SkillIR(
        name="wag_tail",
        profile_id="bittle-standard-biboard-v1-p1s",
        frames=[
            SkillFrame(angles_deg={1: 20}, speed_deg_per_step=4, delay_ms=100),
        ],
    )
    res = validator.validate(ir)
    assert res.valid is False
    assert any("Joint 1 is not installed" in e for e in res.errors)


def test_reject_agent_envelope_violation(validator):
    # Head agent envelope is [-60, 60]. Commanded at 80 deg.
    ir = SkillIR(
        name="extreme_pan",
        profile_id="bittle-standard-biboard-v1-p1s",
        frames=[
            SkillFrame(angles_deg={0: 80}, speed_deg_per_step=4, delay_ms=100),
        ],
    )
    res = validator.validate(ir)
    assert res.valid is False
    assert any("violates agent safety envelope" in e for e in res.errors)


def test_reject_excessive_step_delta(validator):
    # Step jump of 60 deg between adjacent frames exceeds MAX_STEP_DELTA_DEG (45)
    ir = SkillIR(
        name="jerk_move",
        profile_id="bittle-standard-biboard-v1-p1s",
        frames=[
            SkillFrame(angles_deg={0: -30}, speed_deg_per_step=4, delay_ms=100),
            SkillFrame(angles_deg={0: 30}, speed_deg_per_step=4, delay_ms=100),
        ],
    )
    res = validator.validate(ir)
    assert res.valid is False
    assert any("exceeds max step limit" in e for e in res.errors)


def test_reject_cumulative_travel_budget_overflow(validator):
    # Loop count = 20 with 40 deg oscillation = 800 deg cumulative travel > 720
    frames = [
        SkillFrame(angles_deg={0: 20}, speed_deg_per_step=4, delay_ms=50),
        SkillFrame(angles_deg={0: -20}, speed_deg_per_step=4, delay_ms=50),
    ]
    ir = SkillIR(
        name="burnout_nod",
        profile_id="bittle-standard-biboard-v1-p1s",
        loop_count=20,
        frames=frames,
    )
    res = validator.validate(ir)
    assert res.valid is False
    assert any("exceeds safety wear budget" in e for e in res.errors)


def test_reject_infinite_loop():
    with pytest.raises(ValueError) as exc:
        SkillIR(
            name="infinite_gait",
            profile_id="bittle-standard-biboard-v1-p1s",
            loop_count=-1,
            frames=[SkillFrame(angles_deg={0: 0})],
        )
    err_str = str(exc.value)
    assert "greater than or equal to 1" in err_str or "Infinite or negative loops" in err_str
