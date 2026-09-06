"""Typed Intermediate Representation (IR) and Manifest schemas for Bittle skills."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class SkillKind(StrEnum):
    POSTURE = "posture"
    GAIT = "gait"
    BEHAVIOR = "behavior"


class SkillFrame(BaseModel):
    angles_deg: dict[int, int] = Field(
        description="Joint angles in degrees for each commanded joint"
    )
    speed_deg_per_step: int = Field(
        default=4,
        ge=1,
        le=127,
        description="Interpolation speed in deg per step (1=slowest/smoothest, 127=fastest)",
    )
    delay_ms: int = Field(
        default=100,
        ge=0,
        le=5000,
        description="Pause duration after reaching frame target",
    )
    trigger_axis: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="Optional IMU trigger axis (0=none, 1=pitch, 2=roll, 3=yaw)",
    )
    trigger_angle: int | None = Field(
        default=None,
        description="Optional trigger angle threshold",
    )


class SkillIR(BaseModel):
    name: str = Field(min_length=1, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")
    profile_id: str
    kind: SkillKind = SkillKind.BEHAVIOR
    entry_posture: str = "balance"
    exit_posture: str = "balance"
    max_executions: int = Field(default=1, ge=1, le=10)
    loop_count: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Repetition count. Negative/infinite (-1) is forbidden.",
    )
    frames: list[SkillFrame] = Field(min_length=1, max_length=64)

    @field_validator("loop_count")
    @classmethod
    def validate_no_infinite_loops(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Infinite or negative loops (-1) are strictly forbidden for safety")
        return v


class MotionBudget(BaseModel):
    max_delta_deg: float
    total_travel_deg: dict[int, float]
    max_travel_single_joint_deg: float
    direction_reversals: dict[int, int]
    estimated_duration_sec: float
    is_bounded: bool = True


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    budget: MotionBudget | None = None


class SkillManifest(BaseModel):
    ir: SkillIR
    ir_hash: str
    compiled_payload: str  # Hex string of compiled bytes
    payload_hash: str      # SHA-256 of compiled raw bytes
    compiler_version: str = "1.0.0"
    validation_timestamp: str
    simulated: bool = False
    approved_by: str | None = None
    approved_at: str | None = None
    promoted: bool = False
    status: str = "draft"  # draft | validated | simulated | canaried | approved | revoked
