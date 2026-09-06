"""Simulated Bittle -- the default target.

Deliberately a kinematic model, not a physics engine. It tracks joint state,
enforces the same limits, and models slew time so command pacing is realistic.
That is enough to exercise the full command path end to end without risking
hardware, which is the point of a default target.

What it does NOT do: contact forces, balance, falling over. A `PyBulletBackend`
can implement the same interface later; see PLAN.md §3 for why that is out of
scope now rather than half-built here.
"""

from __future__ import annotations

import asyncio
import time

from .. import joints, protocol, skills
from ..motion.builtin_library import BUILTIN_MOVESETS
from .base import Backend, CommandResult

#: Approximate servo slew, degrees per second. Used to report a plausible
#: settle time rather than pretending motion is instantaneous.
SLEW_DEG_PER_SEC = 320.0


class SimBackend(Backend):
    name = "sim"
    is_real_hardware = False

    def __init__(self) -> None:
        self._angles: dict[int, int] = dict(joints.DEFAULT_POSE)
        self._connected = False
        self._last_skill: str | None = "balance"
        self._resting = False
        self._history: list[dict] = []
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    def _record(self, wire: bytes, kind: str) -> None:
        self._history.append({"at": time.time(), "kind": kind, "wire": wire.decode("latin-1")})
        del self._history[:-200]

    async def move_joints(
        self, pairs: list[tuple[int, int]], *, simultaneous: bool = True
    ) -> CommandResult:
        wire = protocol.encode_joint_move(pairs, simultaneous=simultaneous)
        async with self._lock:
            travel = 0.0
            for index, angle in pairs:
                delta = abs(angle - self._angles.get(index, 0))
                # Simultaneous motion settles with the slowest joint; sequential
                # motion accumulates, which is part of why it is not the default.
                travel = max(travel, delta) if simultaneous else travel + delta
                self._angles[index] = angle
            self._resting = False
            self._last_skill = None
            self._record(wire, "move")
        settle = round(travel / SLEW_DEG_PER_SEC, 3)
        return CommandResult(
            ok=True,
            sent=wire.decode("latin-1"),
            response="sim:ok",
            meta={"settle_seconds": settle, "simultaneous": simultaneous},
        )

    async def run_skill(self, skill_name: str) -> CommandResult:
        wire = protocol.encode_skill(skill_name)
        skill = skills.BY_NAME.get(skill_name)
        async with self._lock:
            self._last_skill = skill_name
            self._resting = skill_name == "rest"
            if skill_name in BUILTIN_MOVESETS:
                frames = BUILTIN_MOVESETS[skill_name].get("frames", [])
                if frames:
                    self._angles.update(frames[-1]["angles"])
                if skill_name in ("balance", "stand", "up"):
                    self._resting = False
            elif skill_name in ("balance", "stand", "up"):
                self._angles = dict(joints.STAND_POSE)
                self._resting = False
            elif self._resting:
                self._angles = dict(joints.REST_POSE)
            self._record(wire, "skill")
        return CommandResult(
            ok=True,
            sent=wire.decode("latin-1"),
            response="sim:ok",
            meta={"skill": skill_name, "locomotes": bool(skill and skill.locomotes)},
        )

    async def rest(self) -> CommandResult:
        wire = protocol.encode_rest()
        async with self._lock:
            self._angles = dict(joints.REST_POSE)
            self._resting = True
            self._last_skill = None
            self._record(wire, "rest")
        return CommandResult(ok=True, sent=wire.decode("latin-1"), response="sim:ok")

    async def read_joints(self) -> dict[int, int]:
        return dict(self._angles)

    async def status(self) -> dict:
        return {
            "backend": self.name,
            "real_hardware": False,
            "connected": self._connected,
            "resting": self._resting,
            "last_skill": self._last_skill,
            "joints": dict(self._angles),
            "recent_commands": self._history[-10:],
        }
