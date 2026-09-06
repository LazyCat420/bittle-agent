"""Versioned Hardware Profiles for Petoi Bittle Quadruped.

Replaces single static assumptions with explicit, verified robot hardware profiles:
- Bittle vs Bittle X
- NyBoard vs BiBoard
- P1S alloy vs P1L plastic servos
- Installed joints (9-servo standard vs 10-servo tail/arm equipped)
- 4-tier joint safety envelopes: firmware, transport, tested, and agent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILES_DIR = Path(__file__).resolve().parent.parent / "hardware_profiles"


@dataclass(frozen=True)
class JointEnvelope:
    firmware_min: int
    firmware_max: int
    transport_min: int
    transport_max: int
    tested_min: int
    tested_max: int
    agent_min: int
    agent_max: int

    @classmethod
    def from_dict(cls, data: dict[str, list[int]]) -> JointEnvelope:
        fw = data.get("firmware", [-120, 120])
        tr = data.get("transport", [-128, 127])
        te = data.get("tested", [-90, 90])
        ag = data.get("agent", [-60, 60])
        return cls(
            firmware_min=fw[0], firmware_max=fw[1],
            transport_min=tr[0], transport_max=tr[1],
            tested_min=te[0], tested_max=te[1],
            agent_min=ag[0], agent_max=ag[1],
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "firmware": [self.firmware_min, self.firmware_max],
            "transport": [self.transport_min, self.transport_max],
            "tested": [self.tested_min, self.tested_max],
            "agent": [self.agent_min, self.agent_max],
        }


@dataclass(frozen=True)
class HardwareProfile:
    profile_id: str
    robot_model: str
    board: str
    firmware_repo: str
    firmware_commit: str
    firmware_version: str
    servo_model: str
    feedback_capable: bool
    installed_joints: tuple[int, ...]
    calibration_offsets: dict[int, int]
    envelopes: dict[int, JointEnvelope]

    def is_installed(self, joint_index: int) -> bool:
        return joint_index in self.installed_joints

    def get_envelope(self, joint_index: int) -> JointEnvelope:
        if joint_index not in self.envelopes:
            raise KeyError(f"Joint {joint_index} not in profile {self.profile_id}")
        return self.envelopes[joint_index]

    def profile_hash(self) -> str:
        canonical = {
            "profile_id": self.profile_id,
            "robot_model": self.robot_model,
            "board": self.board,
            "firmware_commit": self.firmware_commit,
            "installed_joints": list(self.installed_joints),
            "calibration_offsets": {str(k): v for k, v in sorted(self.calibration_offsets.items())},
            "envelopes": {str(k): v.to_dict() for k, v in sorted(self.envelopes.items())},
        }
        encoded = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        envelopes = {
            int(k): JointEnvelope.from_dict(v)
            for k, v in data.get("envelopes", {}).items()
        }
        calib = {
            int(k): int(v)
            for k, v in data.get("calibration_offsets", {}).items()
        }
        return cls(
            profile_id=str(data["profile_id"]),
            robot_model=str(data.get("robot_model", "BITTLE")),
            board=str(data.get("board", "BiBoard_V1_0")),
            firmware_repo=str(data.get("firmware_repo", "PetoiCamp/OpenCatEsp32")),
            firmware_commit=str(data.get("firmware_commit", "unknown")),
            firmware_version=str(data.get("firmware_version", "2.1")),
            servo_model=str(data.get("servo_model", "P1S")),
            feedback_capable=bool(data.get("feedback_capable", False)),
            installed_joints=tuple(sorted(int(i) for i in data.get("installed_joints", []))),
            calibration_offsets=calib,
            envelopes=envelopes,
        )


class ProfileRegistry:
    """Loads and caches hardware profiles from disk."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory or PROFILES_DIR
        self._profiles: dict[str, HardwareProfile] = {}
        self.load_all()

    def load_all(self) -> None:
        candidates = [
            self.directory,
            Path("/app/hardware_profiles"),
            Path(__file__).resolve().parent.parent / "hardware_profiles",
            Path.cwd() / "hardware_profiles",
        ]
        loaded_any = False
        for d in candidates:
            if d.exists() and d.is_dir():
                for path in d.glob("*.json"):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        profile = HardwareProfile.from_dict(data)
                        self._profiles[profile.profile_id] = profile
                        loaded_any = True
                    except Exception:
                        continue
            if loaded_any:
                break

        if not self._profiles:
            fallback = self._build_embedded_fallback()
            self._profiles[fallback.profile_id] = fallback

    @staticmethod
    def _build_embedded_fallback() -> HardwareProfile:
        env_defaults = {
            0: JointEnvelope(-120, 120, -128, 127, -90, 90, -60, 60),
            1: JointEnvelope(-85, 85, -128, 127, -60, 60, -45, 45),
            8: JointEnvelope(-200, 80, -128, 80, -115, 65, -80, 50),
            9: JointEnvelope(-200, 80, -128, 80, -115, 65, -80, 50),
            10: JointEnvelope(-80, 200, -80, 127, -65, 115, -50, 80),
            11: JointEnvelope(-80, 200, -80, 127, -65, 115, -50, 80),
            12: JointEnvelope(-80, 200, -80, 127, -65, 115, 30, 95),
            13: JointEnvelope(-80, 200, -80, 127, -65, 115, 30, 95),
            14: JointEnvelope(-80, 200, -80, 127, -65, 115, 30, 95),
            15: JointEnvelope(-80, 200, -80, 127, -65, 115, 30, 95),
        }
        return HardwareProfile(
            profile_id="bittle-standard-biboard-v1-p1s",
            robot_model="BITTLE",
            board="BiBoard_V1_0",
            firmware_repo="PetoiCamp/OpenCatEsp32",
            firmware_commit="e8d6411",
            firmware_version="2.1",
            servo_model="P1S",
            feedback_capable=True,
            installed_joints=frozenset({0, 8, 9, 10, 11, 12, 13, 14, 15}),
            calibration_offsets={i: 0 for i in range(16)},
            envelopes=env_defaults,
        )

    def get(self, profile_id: str) -> HardwareProfile:
        if profile_id not in self._profiles:
            raise KeyError(f"Hardware profile {profile_id!r} not found in registry")
        return self._profiles[profile_id]

    def default(self) -> HardwareProfile:
        # Default to standard BiBoard (Bittle X) profile if available, else first
        if "bittle-standard-biboard-v1-p1s" in self._profiles:
            return self._profiles["bittle-standard-biboard-v1-p1s"]
        if self._profiles:
            return next(iter(self._profiles.values()))
        raise RuntimeError("No hardware profiles loaded in ProfileRegistry")

    def list_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "profile_id": p.profile_id,
                "robot_model": p.robot_model,
                "board": p.board,
                "installed_joints": list(p.installed_joints),
                "feedback_capable": p.feedback_capable,
                "profile_hash": p.profile_hash()[:12],
            }
            for p in self._profiles.values()
        ]


_GLOBAL_REGISTRY: ProfileRegistry | None = None


def get_registry() -> ProfileRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ProfileRegistry()
    return _GLOBAL_REGISTRY


def get_default_profile() -> HardwareProfile:
    return get_registry().default()
