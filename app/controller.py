"""Controller -- the only path from an API request to a backend.

Every public method here validates first and actuates second. Routes must not
hold backend references directly; if they did, a future route could skip the
validator, and the safety guarantee would quietly become a convention.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import joints, protocol
from .backends.base import Backend, CommandResult
from .backends.sim import SimBackend
from .config import Settings
from .motion import get_lifecycle
from .profiles import HardwareProfile, get_registry
from .safety import SafetyError, SafetyValidator, ValidatedMove

logger = logging.getLogger(__name__)


class TargetUnavailable(RuntimeError):
    pass


class Controller:
    def __init__(self, settings: Settings, profile: HardwareProfile | None = None):
        self.settings = settings
        self.profile = profile or joints.get_active_profile()
        self.safety = SafetyValidator(
            rate_per_sec=settings.rate_per_sec,
            burst=settings.rate_burst,
            allow_locomotion=settings.allow_locomotion,
            profile=self.profile,
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

    def set_profile(self, profile_or_id: HardwareProfile | str) -> HardwareProfile:
        prof = joints.set_active_profile(profile_or_id)
        self.profile = prof
        self.safety.profile = prof
        return prof

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
        envelope_tier: str = "transport",
    ) -> tuple[CommandResult, ValidatedMove]:
        validated = self.safety.validate_move(
            angles, simultaneous=simultaneous, envelope_tier=envelope_tier
        )
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

    async def run_approved_skill(
        self,
        manifest_hash: str,
        *,
        target: str = "sim",
        confirm: str | None = None,
    ) -> tuple[CommandResult, Any]:
        lifecycle = get_lifecycle()
        manifest = lifecycle.get_manifest(manifest_hash)

        if target == "real" and not lifecycle.is_callable_on_real(manifest_hash):
            raise TargetUnavailable(
                f"Skill {manifest.ir.name!r} ({manifest_hash[:8]}) has status {manifest.status!r}; "
                "only 'promoted' skills can execute on real hardware"
            )

        backend = self._resolve(target, confirm)

        # Execute frame by frame in backend
        last_res = CommandResult(ok=True, sent=f"skill:{manifest.ir.name}", response="ok")
        for frame in manifest.ir.frames:
            # Send simultaneous move for the frame
            validated = self.safety.validate_move(frame.angles_deg, simultaneous=True)
            last_res = await backend.move_joints(validated.pairs, simultaneous=True)
            if frame.delay_ms > 0:
                await asyncio.sleep(frame.delay_ms / 1000.0)

        return last_res, manifest

    async def estop(self, reason: str = "manual") -> dict:
        """Trip the E-stop and rest every reachable backend immediately (torque release)."""
        self.safety.engage_estop(reason)
        outcomes: dict[str, str] = {}
        for label, backend in (("sim", self._sim), ("real", self._real)):
            if backend is None:
                continue
            try:
                await backend.rest()
                outcomes[label] = "rested"
            except Exception as exc:
                logger.exception("estop rest failed on %s", label)
                outcomes[label] = f"error: {exc}"
        return {"estop": self.safety.estop_status(), "backends": outcomes}

    async def controlled_stop(self, reason: str = "controlled_stop") -> dict:
        """Gracefully transition robot into crouched rest pose, then release torque."""
        self.safety.engage_controlled_stop(reason)
        outcomes: dict[str, str] = {}
        for label, backend in (("sim", self._sim), ("real", self._real)):
            if backend is None:
                continue
            try:
                # Transition legs smoothly to rest pose
                await backend.run_skill("rest")
                outcomes[label] = "crouched_rest"
            except Exception as exc:
                logger.exception("controlled stop failed on %s", label)
                outcomes[label] = f"error: {exc}"
        return {"controlled_stop": self.safety.estop_status(), "backends": outcomes}

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
            "profile": {
                "profile_id": self.profile.profile_id,
                "robot_model": self.profile.robot_model,
                "board": self.profile.board,
                "feedback_capable": self.profile.feedback_capable,
                "installed_joints": list(self.profile.installed_joints),
                "profile_hash": self.profile.profile_hash()[:12],
            },
        }

    async def joint_state(self, target: str = "sim", confirm: str | None = None) -> dict[int, int]:
        backend = self._resolve(target, confirm)
        return await backend.read_joints()

    @staticmethod
    def describe_joints() -> list[dict]:
        active_prof = joints.get_active_profile()
        return [
            {
                "index": j.index,
                "name": j.name,
                "label": j.label,
                "min": j.safe_min,
                "max": j.safe_max,
                "firmware_min": j.fw_min,
                "firmware_max": j.fw_max,
                "tested_min": j.tested_min,
                "tested_max": j.tested_max,
                "agent_min": j.agent_min,
                "agent_max": j.agent_max,
                "wire_clipped": j.wire_clipped,
                "default": joints.DEFAULT_POSE.get(j.index, 0),
                "rest": joints.REST_POSE.get(j.index, 0),
                "installed": active_prof.is_installed(j.index),
            }
            for j in joints.CONTROLLABLE
        ]

    @staticmethod
    def preview(angles: dict[int, int], simultaneous: bool = True) -> str:
        return protocol.encode_joint_move(
            list(angles.items()), simultaneous=simultaneous
        ).decode("latin-1")
