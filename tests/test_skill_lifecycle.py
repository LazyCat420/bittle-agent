"""Tests for Skill Promotion Lifecycle and Gate Enforcement."""

from __future__ import annotations

import pytest

from app.motion.lifecycle import LifecycleError, SkillLifecycleManager
from app.motion.schema import SkillFrame, SkillIR


@pytest.fixture
def lifecycle():
    return SkillLifecycleManager()


@pytest.fixture
def sample_ir():
    return SkillIR(
        name="test_lifecyle_skill",
        profile_id="bittle-standard-biboard-v1-p1s",
        frames=[
            SkillFrame(angles_deg={0: 10}, speed_deg_per_step=4, delay_ms=50),
            SkillFrame(angles_deg={0: 0}, speed_deg_per_step=4, delay_ms=50),
        ],
    )


def test_lifecycle_full_promotion_pipeline(lifecycle, sample_ir):
    # 1. Validate & Compile
    val_res, manifest = lifecycle.validate_and_compile(sample_ir)
    assert val_res.valid is True
    assert manifest is not None
    m_hash = manifest.payload_hash
    assert manifest.status == "validated"

    # Cannot promote directly from validated without simulation and approval
    with pytest.raises(LifecycleError):
        lifecycle.promote(m_hash)

    # 2. Record Simulation
    manifest = lifecycle.record_simulation(m_hash, {"sim_fk": "pass"})
    assert manifest.status == "simulated"
    assert manifest.simulated is True

    # 3. Record Canary
    manifest = lifecycle.record_canary(m_hash, "Ran on stand with feet suspended; smooth motion")
    assert manifest.status == "canaried"

    # 4. Human Approval
    manifest = lifecycle.approve(m_hash, approver="operator_human")
    assert manifest.status == "approved"
    assert manifest.approved_by == "operator_human"

    # 5. Promotion
    assert not lifecycle.is_callable_on_real(m_hash)
    manifest = lifecycle.promote(m_hash)
    assert manifest.status == "promoted"
    assert manifest.promoted is True
    assert lifecycle.is_callable_on_real(m_hash)

    # 6. Revocation
    manifest = lifecycle.revoke(m_hash, reason="re-tested with new battery")
    assert manifest.status == "revoked"
    assert not lifecycle.is_callable_on_real(m_hash)
