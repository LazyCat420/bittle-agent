"""Tests for Skill Compiler and OpenCat Wire Generation."""

from __future__ import annotations

import pytest

from app.motion.compiler import SkillCompiler, SkillCompilerError
from app.motion.schema import SkillFrame, SkillIR, SkillKind


@pytest.fixture
def compiler():
    return SkillCompiler()


def test_compile_generates_valid_signed_manifest(compiler):
    ir = SkillIR(
        name="bow_head",
        profile_id="bittle-standard-biboard-v1-p1s",
        kind=SkillKind.BEHAVIOR,
        frames=[
            SkillFrame(angles_deg={0: 20}, speed_deg_per_step=4, delay_ms=100),
            SkillFrame(angles_deg={0: 0}, speed_deg_per_step=4, delay_ms=100),
        ],
    )
    manifest = compiler.compile(ir)
    assert manifest.ir_hash is not None
    assert len(manifest.ir_hash) == 64
    assert manifest.payload_hash is not None
    assert len(manifest.payload_hash) == 64
    assert manifest.status == "validated"

    # Verify wire payload begins with 'K' and ends with '~'
    payload_bytes = bytes.fromhex(manifest.compiled_payload)
    assert payload_bytes.startswith(b"K")
    assert payload_bytes.endswith(b"~")


def test_compile_deterministic_hash(compiler):
    ir = SkillIR(
        name="test_det",
        profile_id="bittle-standard-biboard-v1-p1s",
        frames=[SkillFrame(angles_deg={0: 10})],
    )
    m1 = compiler.compile(ir)
    m2 = compiler.compile(ir)
    assert m1.payload_hash == m2.payload_hash
    assert m1.ir_hash == m2.ir_hash


def test_compile_invalid_ir_raises_error(compiler):
    # Invalid IR with out-of-bounds angle
    ir = SkillIR(
        name="bad_skill",
        profile_id="bittle-standard-biboard-v1-p1s",
        frames=[SkillFrame(angles_deg={0: 999})],
    )
    with pytest.raises(SkillCompilerError):
        compiler.compile(ir)
