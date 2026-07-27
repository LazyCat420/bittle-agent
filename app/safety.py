"""Mandatory safety layer.

Every motion command passes through `SafetyValidator` before any backend sees
it. There is no bypass path -- the backends are only reachable via the
validator, and the validator fails closed.

Design rule: this module decides, it does not actuate. That keeps it pure and
therefore testable, which is the only reason to believe it works.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum

from . import joints, skills


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
    """Token bucket bounding sustained command rate.

    This is a hardware-protection guard, not an API quota: continuous servo
    thrash is how these joints overheat. Bursts are fine, sustained flooding is
    not, which is exactly a token bucket's shape.
    """

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
    """Clamp, whitelist, rate-limit and E-stop gate for all robot commands."""

    def __init__(
        self,
        *,
        rate_per_sec: float = 8.0,
        burst: int = 16,
        allow_locomotion: bool = True,
    ):
        self._limiter = RateLimiter(rate_per_sec, burst)
        self._estop = False
        self._estop_reason: str | None = None
        self._estop_at: float | None = None
        self._allow_locomotion = allow_locomotion
        self._lock = threading.Lock()

    # ── E-stop ────────────────────────────────────────────────────────────
    # Latching by design. An E-stop that auto-clears is not an E-stop: whatever
    # tripped it is still true until a human says otherwise.

    @property
    def estop_engaged(self) -> bool:
        with self._lock:
            return self._estop

    def engage_estop(self, reason: str = "manual") -> None:
        with self._lock:
            self._estop = True
            self._estop_reason = reason
            self._estop_at = time.time()

    def clear_estop(self) -> None:
        with self._lock:
            self._estop = False
            self._estop_reason = None
            self._estop_at = None

    def estop_status(self) -> dict:
        with self._lock:
            return {
                "engaged": self._estop,
                "reason": self._estop_reason,
                "engaged_at": self._estop_at,
            }

    def _assert_operational(self) -> None:
        if self.estop_engaged:
            raise EStopEngaged()

    def _assert_rate(self, cost: int = 1) -> None:
        if not self._limiter.consume(cost):
            raise SafetyError(
                "rate_limited",
                "command rate exceeded; this limit protects the servos from "
                "sustained thrash",
            )

    # ── Joint motion ──────────────────────────────────────────────────────

    def validate_move(
        self,
        requested: dict[int | str, float],
        *,
        simultaneous: bool = True,
    ) -> ValidatedMove:
        """Validate a joint move. Clamps angles, rejects unknown joints."""
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

            if not joint.used:
                raise SafetyError(
                    "unused_joint",
                    f"joint {joint.index} ({joint.name}) is not populated on Bittle; "
                    "commanding it would drive a servo that isn't there",
                )

            # Check finiteness BEFORE int(round(...)), because round(nan) raises
            # ValueError and int(inf) raises OverflowError -- both would surface
            # as a confusing crash rather than a clean rejection. NaN is the
            # dangerous one: it silently fails every `<` comparison, so an
            # unchecked NaN would sail straight through the clamp below.
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

            applied = max(joint.safe_min, min(joint.safe_max, angle))
            if applied != angle:
                # Distinguish the two causes: "joint_limit" means the hardware
                # itself won't go there; "wire_range" means the firmware would
                # allow it but the signed-char encoding cannot carry it. They
                # have different fixes, so don't collapse them into one reason.
                reason = "wire_range" if joint.fw_min <= angle <= joint.fw_max else "joint_limit"
                adjustments.append(
                    Adjustment(
                        joint_index=joint.index,
                        joint_name=joint.name,
                        requested=angle,
                        applied=applied,
                        reason=reason,
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
            "rate_limit": {
                "tokens_available": round(self._limiter.available, 2),
                "rate_per_sec": self._limiter.rate,
                "burst": self._limiter.burst,
            },
            "locomotion_allowed": self._allow_locomotion,
        }
