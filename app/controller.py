"""Controller -- the only path from an API request to a backend.

Every public method here validates first and actuates second. Routes must not
hold backend references directly; if they did, a future route could skip the
validator, and the safety guarantee would quietly become a convention.
"""

from __future__ import annotations

import asyncio
import logging

from . import joints, protocol
from .backends.base import Backend, CommandResult
from .backends.sim import SimBackend
from .config import Settings
from .safety import SafetyError, SafetyValidator, ValidatedMove

logger = logging.getLogger(__name__)


class TargetUnavailable(RuntimeError):
    pass


class Controller:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.safety = SafetyValidator(
            rate_per_sec=settings.rate_per_sec,
            burst=settings.rate_burst,
            allow_locomotion=settings.allow_locomotion,
        )
        self._sim = SimBackend()
        self._real: Backend | None = None
        self._real_error: str | None = None
        self._start_lock = asyncio.Lock()

    async def startup(self) -> None:
        await self._sim.connect()
        if not self.settings.allow_real_hardware:
            logger.info("real hardware disabled (BITTLE_ALLOW_REAL_HARDWARE not set)")
            return
        # Construct the serial backend lazily and tolerate failure: a missing
        # robot must not stop the service from serving sim traffic.
        from .backends.serial_backend import SerialBackend, SerialUnavailable

        backend = SerialBackend(self.settings.serial_port, self.settings.serial_baud)
        try:
            await backend.connect()
            self._real = backend
            logger.info("real hardware ENABLED on %s", self.settings.serial_port)
        except SerialUnavailable as exc:
            self._real_error = str(exc)
            logger.warning("real hardware unavailable: %s", exc)

    async def shutdown(self) -> None:
        await self._sim.close()
        if self._real is not None:
            await self._real.close()

    # ── Target resolution ─────────────────────────────────────────────────

    def _resolve(self, target: str, confirm: str | None) -> Backend:
        """Pick a backend. `sim` is the default and needs no ceremony."""
        if target == "sim":
            return self._sim
        if target != "real":
            raise TargetUnavailable(f"unknown target {target!r} (use 'sim' or 'real')")

        if not self.settings.allow_real_hardware:
            raise TargetUnavailable(
                "real hardware is disabled; set BITTLE_ALLOW_REAL_HARDWARE=true to enable"
            )
        if self.settings.requires_confirm_token and confirm != self.settings.confirm_token:
            raise TargetUnavailable(
                "real-hardware commands require a valid confirm token"
            )
        if self._real is None:
            raise TargetUnavailable(
                f"serial backend not connected: {self._real_error or 'unknown error'}"
            )
        return self._real

    # ── Commands ──────────────────────────────────────────────────────────

    async def move(
        self,
        angles: dict[int | str, float],
        *,
        target: str = "sim",
        simultaneous: bool = True,
        confirm: str | None = None,
    ) -> tuple[CommandResult, ValidatedMove]:
        validated = self.safety.validate_move(angles, simultaneous=simultaneous)
        backend = self._resolve(target, confirm)
        result = await backend.move_joints(
            validated.pairs, simultaneous=validated.simultaneous
        )
        return result, validated

    async def skill(
        self,
        name: str,
        *,
        target: str = "sim",
        ack_locomotion: bool = False,
        confirm: str | None = None,
    ) -> tuple[CommandResult, object]:
        skill = self.safety.validate_skill(name, ack_locomotion=ack_locomotion)
        backend = self._resolve(target, confirm)
        result = await backend.run_skill(skill.name)
        return result, skill

    async def estop(self, reason: str = "manual") -> dict:
        """Trip the E-stop and rest every reachable backend.

        Order matters: engage the latch FIRST so nothing new is admitted while
        we are still sending rest commands. Also note `rest` deliberately
        bypasses the validator's rate limiter -- an emergency stop that can be
        rate-limited is not an emergency stop.
        """
        self.safety.engage_estop(reason)
        outcomes: dict[str, str] = {}
        for label, backend in (("sim", self._sim), ("real", self._real)):
            if backend is None:
                continue
            try:
                await backend.rest()
                outcomes[label] = "rested"
            except Exception as exc:  # never let one backend block the other
                logger.exception("estop rest failed on %s", label)
                outcomes[label] = f"error: {exc}"
        return {"estop": self.safety.estop_status(), "backends": outcomes}

    def clear_estop(self) -> dict:
        self.safety.clear_estop()
        return self.safety.estop_status()

    async def status(self, target: str = "sim") -> dict:
        backend = self._sim if target != "real" else self._real
        backend_status = await backend.status() if backend is not None else {
            "backend": "real",
            "connected": False,
            "error": self._real_error,
        }
        return {
            "safety": self.safety.status(),
            "target": backend_status,
            "real_hardware_allowed": self.settings.allow_real_hardware,
            "confirm_token_required": self.settings.requires_confirm_token,
        }

    async def joint_state(self, target: str = "sim", confirm: str | None = None) -> dict[int, int]:
        backend = self._resolve(target, confirm)
        return await backend.read_joints()

    @staticmethod
    def describe_joints() -> list[dict]:
        return [
            {
                "index": j.index,
                "name": j.name,
                "label": j.label,
                "min": j.safe_min,
                "max": j.safe_max,
                "firmware_min": j.fw_min,
                "firmware_max": j.fw_max,
                "wire_clipped": j.wire_clipped,
                "rest": joints.REST_POSE.get(j.index, 0),
            }
            for j in joints.CONTROLLABLE
        ]

    @staticmethod
    def preview(angles: dict[int, int], simultaneous: bool = True) -> str:
        return protocol.encode_joint_move(
            list(angles.items()), simultaneous=simultaneous
        ).decode("latin-1")
