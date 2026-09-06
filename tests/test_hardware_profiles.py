"""Tests for Versioned Hardware Profiles and 4-Tier Joint Envelopes."""

from __future__ import annotations

import pytest

from app.joints import (
    CONTROLLABLE_INDICES,
    get_active_profile,
    resolve,
    set_active_profile,
)
from app.profiles import HardwareProfile, get_default_profile, get_registry


def test_registry_loads_checked_in_profiles():
    reg = get_registry()
    profiles = reg.list_profiles()
    assert len(profiles) >= 3
    profile_ids = [p["profile_id"] for p in profiles]
    assert "bittle-standard-biboard-v1-p1s" in profile_ids
    assert "bittle-standard-nyboard-v1-p1s" in profile_ids
    assert "bittle-tail-installed-nyboard-v1-p1s" in profile_ids


def test_standard_profile_has_nine_joints_excluding_tail():
    p = get_registry().get("bittle-standard-biboard-v1-p1s")
    assert len(p.installed_joints) == 9
    assert 0 in p.installed_joints
    assert 1 not in p.installed_joints  # Tail must NOT be installed on standard Bittle
    for leg_joint in range(8, 16):
        assert leg_joint in p.installed_joints


def test_tail_installed_profile_has_ten_joints():
    p = get_registry().get("bittle-tail-installed-nyboard-v1-p1s")
    assert len(p.installed_joints) == 10
    assert 1 in p.installed_joints


def test_four_tier_envelopes():
    p = get_registry().get("bittle-standard-biboard-v1-p1s")
    env_head = p.get_envelope(0)
    assert env_head.firmware_min == -120
    assert env_head.firmware_max == 120
    assert env_head.tested_min == -90
    assert env_head.tested_max == 90
    assert env_head.agent_min == -60
    assert env_head.agent_max == 60

    env_knee = p.get_envelope(12)
    assert env_knee.firmware_min == -80
    assert env_knee.firmware_max == 200
    assert env_knee.agent_min == 30
    assert env_knee.agent_max == 95


def test_profile_hash_is_deterministic():
    p1 = get_registry().get("bittle-standard-biboard-v1-p1s")
    h1 = p1.profile_hash()
    h2 = p1.profile_hash()
    assert len(h1) == 64
    assert h1 == h2


def test_dynamic_profile_switching():
    try:
        # Switch to tail profile
        set_active_profile("bittle-tail-installed-nyboard-v1-p1s")
        assert resolve(1).used is True
        from app.joints import get_controllable_indices
        assert 1 in get_controllable_indices()
    finally:
        # Always switch back to standard profile so subsequent tests are clean
        set_active_profile("bittle-standard-biboard-v1-p1s")

    from app.joints import get_controllable_indices
    assert resolve(1).used is False
    assert 1 not in get_controllable_indices()
