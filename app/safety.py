"""Mandatory safety layer for Petoi Bittle.

Every motion command passes through `SafetyValidator` before any backend sees
it. There is no bypass path -- the backends are only reachable via the
validator, and the validator fails closed.

Supports multi-envelope limits:
- firmware: Raw angleLimit acceptance from OpenCat.h
- transport: Signed char boundary on wire (-128..127)
- tested: Physical mechanical clearance limits
- agent: Conservative operational envelope for autonomous LLM generation

Supports two-stage emergency / safe stop policies:
- controlled_stop: smooth transition to rest pose before torque release
- estop: immediate torque cut ('d') and latching motion lock.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum

from . import joints, skills
from .profiles import HardwareProfile


class Decision(StrEnum):
    ALLOW = "allow"
    CLAMP = "clamp"
    REJECT = "reject"


class SafetyError(Exception):
    """Command refused. Carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class EStopEngaged(SafetyError):
    def __init__(self, detail: str = "E-stop is engaged; clear it before moving"):
        super().__init__("estop_engaged", detail)


@dataclass
class Adjustment:
    """A value the validator changed rather than rejected."""

    joint_index: int
    joint_name: str
    requested: int
    applied: int
    reason: str


@dataclass
class ValidatedMove:
    pairs: list[tuple[int, int]]
    adjustments: list[Adjustment] = field(default_factory=list)
    simultaneous: bool = True

    @property
    def clamped(self) -> bool:
        return bool(self.adjustments)


class RateLimiter:
    """Token bucket bounding sustained command rate."""

    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = rate_per_sec
        self.burst = burst
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        with self._lock:
            now = time.monotonic()
            return min(self.burst, self._tokens + (now - self._updated) * self.rate)


class SafetyValidator:
    """Clamp, whitelist, rate-limit, and E-stop gate for all robot commands."""

    def __init__(
        self,
        *,
        rate_per_sec: float = 8.0,
        burst: int = 16,
        allow_locomotion: bool = True,
        profile: HardwareProfile | None = None,
    ):
        self._limiter = RateLimiter(rate_per_sec, burst)
        self._estop = False
        self._estop_reason: str | None = None
        self._estop_at: float | None = None
        self._controlled_stop = False
        self._allow_locomotion = allow_locomotion
        self.profile = profile or joints.get_active_profile()
        self._lock = threading.Lock()

    # ── E-stop & Controlled Stop ──────────────────────────────────────────

    @property
    def estop_engaged(self) -> bool:
        with self._lock:
            return self._estop

    @property
    def controlled_stop_active(self) -> bool:
        with self._lock:
            return self._controlled_stop

    def engage_estop(self, reason: str = "manual") -> None:
        with self._lock:
            self._estop = True
            self._estop_reason = reason
            self._estop_at = time.time()
            self._controlled_stop = False

    def engage_controlled_stop(self, reason: str = "controlled_stop") -> None:
        with self._lock:
            self._controlled_stop = True
            self._estop_reason = reason
            self._estop_at = time.time()

    def clear_estop(self) -> None:
        with self._lock:
            self._estop = False
            self._controlled_stop = False
            self._estop_reason = None
            self._estop_at = None

    def estop_status(self) -> dict:
        with self._lock:
            return {
                "engaged": self._estop,
                "controlled_stop": self._controlled_stop,
                "reason": self._estop_reason,
                "engaged_at": self._estop_at,
            }

    def _assert_operational(self) -> None:
        if self.estop_engaged or self.controlled_stop_active:
            raise EStopEngaged()

    def _assert_rate(self, cost: int = 1) -> None:
        if not self._limiter.consume(cost):
            raise SafetyError(
                "rate_limited",
                "command rate exceeded; this limit protects the servos from sustained thrash",
            )

    # ── Joint motion ──────────────────────────────────────────────────────

    def validate_move(
        self,
        requested: dict[int | str, float],
        *,
        simultaneous: bool = True,
        envelope_tier: str = "transport",
    ) -> ValidatedMove:
        """Validate a joint move. Clamps angles to requested envelope tier, rejects uninstalled joints."""
        self._assert_operational()

        if not requested:
            raise SafetyError("empty_command", "no joints specified")

        pairs: list[tuple[int, int]] = []
        adjustments: list[Adjustment] = []

        for raw_joint, raw_angle in requested.items():
            try:
                joint = joints.resolve(raw_joint)
            except KeyError as exc:
                raise SafetyError("unknown_joint", str(exc)) from exc

            # Enforce profile installed joints
            if not self.profile.is_installed(joint.index):
                raise SafetyError(
                    "unused_joint",
                    f"joint {joint.index} ({joint.name}) is not populated on Bittle ({self.profile.profile_id}); "
                    "commanding it would drive a servo that isn't there",
                )

            try:
                angle_f = float(raw_angle)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "invalid_angle", f"angle for {joint.name} is not a number: {raw_angle!r}"
                ) from exc
            if not math.isfinite(angle_f):
                raise SafetyError(
                    "invalid_angle", f"angle for {joint.name} is not finite: {raw_angle!r}"
                )
            angle = int(round(angle_f))

            # Select envelope tier bounds
            if envelope_tier == "agent":
                min_b, max_b = joint.agent_min, joint.agent_max
                clamp_reason = "agent_envelope"
            elif envelope_tier == "tested":
                min_b, max_b = joint.tested_min, joint.tested_max
                clamp_reason = "tested_clearance"
            else:
                min_b, max_b = joint.safe_min, joint.safe_max
                clamp_reason = "wire_range" if joint.fw_min <= angle <= joint.fw_max else "joint_limit"

            applied = max(min_b, min(max_b, angle))
            if applied != angle:
                adjustments.append(
                    Adjustment(
                        joint_index=joint.index,
                        joint_name=joint.name,
                        requested=angle,
                        applied=applied,
                        reason=clamp_reason,
                    )
                )
            pairs.append((joint.index, applied))

        self._assert_rate(cost=1)
        return ValidatedMove(pairs=pairs, adjustments=adjustments, simultaneous=simultaneous)

    # ── Skills ────────────────────────────────────────────────────────────

    def validate_skill(self, name: str, *, ack_locomotion: bool = False) -> skills.Skill:
        self._assert_operational()
        try:
            skill = skills.resolve(name)
        except KeyError as exc:
            raise SafetyError("unknown_skill", str(exc)) from exc

        if skill.locomotes:
            if not self._allow_locomotion:
                raise SafetyError(
                    "locomotion_disabled",
                    f"{skill.token} moves the robot across the floor and locomotion "
                    "is disabled for this deployment",
                )
            if not ack_locomotion:
                raise SafetyError(
                    "locomotion_unacked",
                    f"{skill.token} drives the robot across the surface it is standing "
                    "on. Re-send with ack_locomotion=true once the area is clear.",
                )

        self._assert_rate(cost=1)
        return skill

    # ── Introspection ─────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "estop": self.estop_status(),
            "profile": {
                "profile_id": self.profile.profile_id,
                "robot_model": self.profile.robot_model,
                "installed_joints": list(self.profile.installed_joints),
                "profile_hash": self.profile.profile_hash()[:12],
            },
            "rate_limit": {
                "tokens_available": round(self._limiter.available, 2),
                "rate_per_sec": self._limiter.rate,
                "burst": self._limiter.burst,
            },
            "locomotion_allowed": self._allow_locomotion,
        }
