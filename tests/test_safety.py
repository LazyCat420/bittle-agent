"""Safety tests.

These assert the DANGEROUS cases, not the happy path. A test suite that only
proves "sit works" tells you nothing about whether the robot is protected.
"""

from __future__ import annotations

import math

import pytest

from app import joints, protocol, skills
from app.protocol import EncodingError
from app.safety import EStopEngaged, SafetyError, SafetyValidator


@pytest.fixture
def validator() -> SafetyValidator:
    # Generous rate so limiting doesn't mask the behaviour under test.
    return SafetyValidator(rate_per_sec=1000, burst=1000)


# ── The signed-char hazard (the headline finding) ──────────────────────────


def test_firmware_legal_angle_beyond_wire_range_is_clamped_not_wrapped(validator):
    """Knee 12 permits +200 per firmware, but binary packing wraps it to -56.

    -56 is not a harmless garble: it drives the joint hard the OTHER way. The
    clamp must hold it at the wire ceiling instead.
    """
    knee = joints.BY_INDEX[12]
    assert knee.fw_max == 200, "firmware table changed; re-verify against OpenCat.h"

    move = validator.validate_move({12: 200})
    (_, applied), = move.pairs

    assert applied == 127
    assert applied != -56
    assert move.clamped
    assert move.adjustments[0].reason == "wire_range"


def test_binary_encoder_refuses_out_of_range_rather_than_wrapping():
    with pytest.raises(EncodingError):
        protocol.encode_binary("L", [200])


def test_ascii_encoding_is_used_for_moves_so_large_angles_survive():
    """ASCII has no signed-char ceiling -- that's why moves use `i`, not `I`."""
    wire = protocol.encode_joint_move([(12, 120)], simultaneous=True)
    assert wire == b"i12 120 \n"


# ── Per-joint limits are asymmetric; a global clamp would be wrong ─────────


def test_tail_limit_is_not_widened_to_a_global_range(validator):
    """Tail is +/-85. A naive global +/-125 would over-rotate it."""
    move = validator.validate_move({1: 120})
    (_, applied), = move.pairs
    assert applied == 85


def test_head_allows_more_than_the_tail(validator):
    move = validator.validate_move({0: 120})
    (_, applied), = move.pairs
    assert applied == 120


def test_negative_shoulder_limit_clamps_to_wire_floor(validator):
    """Shoulder 8 is {-200, 80}; -200 is below the signed-char floor."""
    move = validator.validate_move({8: -200})
    (_, applied), = move.pairs
    assert applied == -128
    assert move.adjustments[0].reason == "wire_range"


# ── Rejections ────────────────────────────────────────────────────────────


def test_unknown_joint_index_is_rejected(validator):
    with pytest.raises(SafetyError) as exc:
        validator.validate_move({99: 0})
    assert exc.value.reason == "unknown_joint"


def test_unused_joint_slot_is_rejected(validator):
    """Slots 4-7 exist for Nybble parity; Bittle has no servo there."""
    with pytest.raises(SafetyError) as exc:
        validator.validate_move({5: 0})
    assert exc.value.reason == "unused_joint"


def test_nan_angle_is_rejected_and_does_not_slip_through_the_clamp(validator):
    """NaN fails every `<` comparison, so an unchecked NaN passes a min/max clamp."""
    with pytest.raises(SafetyError) as exc:
        validator.validate_move({8: math.nan})
    assert exc.value.reason == "invalid_angle"


def test_infinite_angle_is_rejected(validator):
    with pytest.raises(SafetyError) as exc:
        validator.validate_move({8: math.inf})
    assert exc.value.reason == "invalid_angle"


def test_unknown_skill_is_rejected_rather_than_silently_ignored(validator):
    """Firmware ignores unknown `k` tokens, which looks like a hang. Fail loud."""
    with pytest.raises(SafetyError) as exc:
        validator.validate_skill("moonwalk")
    assert exc.value.reason == "unknown_skill"


def test_known_skill_resolves_with_or_without_k_prefix(validator):
    assert validator.validate_skill("sit").token == "ksit"
    assert validator.validate_skill("ksit").token == "ksit"


# ── Locomotion gating ─────────────────────────────────────────────────────


def test_gait_requires_explicit_ack(validator):
    with pytest.raises(SafetyError) as exc:
        validator.validate_skill("wkF")
    assert exc.value.reason == "locomotion_unacked"


def test_gait_allowed_once_acked(validator):
    assert validator.validate_skill("wkF", ack_locomotion=True).name == "wkF"


def test_posture_needs_no_ack(validator):
    assert validator.validate_skill("sit").name == "sit"


def test_locomotion_can_be_disabled_entirely():
    v = SafetyValidator(rate_per_sec=1000, burst=1000, allow_locomotion=False)
    with pytest.raises(SafetyError) as exc:
        v.validate_skill("wkF", ack_locomotion=True)
    assert exc.value.reason == "locomotion_disabled"


# ── E-stop ────────────────────────────────────────────────────────────────


def test_estop_blocks_moves_and_skills(validator):
    validator.engage_estop("test")
    with pytest.raises(EStopEngaged):
        validator.validate_move({8: 0})
    with pytest.raises(EStopEngaged):
        validator.validate_skill("sit")


def test_estop_latches_until_explicitly_cleared(validator):
    validator.engage_estop("test")
    assert validator.estop_engaged
    validator.validate_move  # no implicit expiry path exists
    validator.clear_estop()
    assert not validator.estop_engaged
    assert validator.validate_move({8: 0}).pairs == [(8, 0)]


# ── Rate limiting ─────────────────────────────────────────────────────────


def test_sustained_flooding_is_rate_limited():
    v = SafetyValidator(rate_per_sec=1, burst=3)
    for _ in range(3):
        v.validate_move({8: 0})
    with pytest.raises(SafetyError) as exc:
        v.validate_move({8: 0})
    assert exc.value.reason == "rate_limited"


# ── Encoding details ──────────────────────────────────────────────────────


def test_simultaneous_and_sequential_use_different_tokens():
    assert protocol.encode_joint_move([(8, 10)], simultaneous=True).startswith(b"i")
    assert protocol.encode_joint_move([(8, 10)], simultaneous=False).startswith(b"m")


def test_skill_encoding_matches_opencat_format():
    assert protocol.encode_skill("sit") == b"ksit\n"
    assert protocol.encode_skill("wkF") == b"kwkF\n"


def test_rest_token_is_d():
    assert protocol.encode_rest() == b"d\n"


def test_all_skill_tokens_are_lowercase_k_prefixed():
    for skill in skills.ALL:
        assert skill.token.startswith("k")
