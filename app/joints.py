"""Bittle joint map, 4-tier angle envelopes, and hardware profile integration.

Every constant here is transcribed from OpenCat firmware:
  - angleLimit / middleShift : src/OpenCat.h / src/OpenCatEsp32.h
  - DOF = 16, WALKING_DOF = 8

Supports 4-tier joint envelopes:
  1. firmware_min / firmware_max : Firmware angle acceptance
  2. transport_min / transport_max : Transport wire boundary (signed char -128..127)
  3. tested_min / tested_max : Tested physical mechanical clearance limits
  4. agent_min / agent_max : Conservative operational envelope for autonomous LLM generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .profiles import HardwareProfile, JointEnvelope, get_default_profile, get_registry

DOF = 16
WALKING_DOF = 8

WIRE_MIN = -128
WIRE_MAX = 127


@dataclass(frozen=True)
class Joint:
    index: int
    name: str
    label: str
    fw_min: int
    fw_max: int
    used: bool = True
    tested_min: int = -90
    tested_max: int = 90
    agent_min: int = -60
    agent_max: int = 60

    @property
    def safe_min(self) -> int:
        return max(self.fw_min, WIRE_MIN)

    @property
    def safe_max(self) -> int:
        return min(self.fw_max, WIRE_MAX)

    @property
    def transport_min(self) -> int:
        return WIRE_MIN

    @property
    def transport_max(self) -> int:
        return WIRE_MAX

    @property
    def wire_clipped(self) -> bool:
        return self.fw_min < WIRE_MIN or self.fw_max > WIRE_MAX


# Base raw joints table from OpenCat.h
_BASE_DEFINITIONS: tuple[tuple[int, str, str, int, int], ...] = (
    (0, "head", "Head pan", -120, 120),
    (1, "tail", "Tail", -85, 85),
    (2, "reserved_2", "Reserved", -120, 120),
    (3, "reserved_3", "Reserved", -120, 120),
    (4, "unused_4", "Unused", -90, 60),
    (5, "unused_5", "Unused", -90, 60),
    (6, "unused_6", "Unused", -90, 90),
    (7, "unused_7", "Unused", -90, 90),
    (8, "shoulder_front_left", "Shoulder front-left", -200, 80),
    (9, "shoulder_front_right", "Shoulder front-right", -200, 80),
    (10, "shoulder_back_right", "Shoulder back-right", -80, 200),
    (11, "shoulder_back_left", "Shoulder back-left", -80, 200),
    (12, "knee_front_left", "Knee front-left", -80, 200),
    (13, "knee_front_right", "Knee front-right", -80, 200),
    (14, "knee_back_right", "Knee back-right", -80, 200),
    (15, "knee_back_left", "Knee back-left", -80, 200),
)


def _build_joints(profile: HardwareProfile) -> tuple[Joint, ...]:
    result: list[Joint] = []
    for idx, name, label, fw_min, fw_max in _BASE_DEFINITIONS:
        installed = profile.is_installed(idx)
        if idx in profile.envelopes:
            env = profile.get_envelope(idx)
            te_min, te_max = env.tested_min, env.tested_max
            ag_min, ag_max = env.agent_min, env.agent_max
        else:
            te_min, te_max = -90, 90
            ag_min, ag_max = -60, 60

        result.append(
            Joint(
                index=idx,
                name=name,
                label=label,
                fw_min=fw_min,
                fw_max=fw_max,
                used=installed,
                tested_min=te_min,
                tested_max=te_max,
                agent_min=ag_min,
                agent_max=ag_max,
            )
        )
    return tuple(result)


_ACTIVE_PROFILE: HardwareProfile = get_default_profile()
_JOINTS: tuple[Joint, ...] = _build_joints(_ACTIVE_PROFILE)
BY_INDEX: dict[int, Joint] = {j.index: j for j in _JOINTS}
BY_NAME: dict[str, Joint] = {j.name: j for j in _JOINTS}
CONTROLLABLE: tuple[Joint, ...] = tuple(j for j in _JOINTS if j.used)
CONTROLLABLE_INDICES: frozenset[int] = frozenset(j.index for j in CONTROLLABLE)


def set_active_profile(profile_or_id: HardwareProfile | str) -> HardwareProfile:
    """Switch the active hardware profile dynamically."""
    global _ACTIVE_PROFILE, _JOINTS, BY_INDEX, BY_NAME, CONTROLLABLE, CONTROLLABLE_INDICES
    if isinstance(profile_or_id, str):
        profile = get_registry().get(profile_or_id)
    else:
        profile = profile_or_id

    _ACTIVE_PROFILE = profile
    _JOINTS = _build_joints(profile)
    BY_INDEX = {j.index: j for j in _JOINTS}
    BY_NAME = {j.name: j for j in _JOINTS}
    CONTROLLABLE = tuple(j for j in _JOINTS if j.used)
    CONTROLLABLE_INDICES = frozenset(j.index for j in CONTROLLABLE)
    return profile


def get_active_profile() -> HardwareProfile:
    return _ACTIVE_PROFILE


def get_controllable_indices() -> frozenset[int]:
    return CONTROLLABLE_INDICES


#: Authentic standing posture where all 4 legs support the robot upright.
STAND_POSE: dict[int, int] = {
    0: 0,   # Head pan centered
    1: 0,   # Tail centered (if installed)
    8: -45, 9: -45,    # Front shoulders pitched downward
    10: -45, 11: -45,  # Rear shoulders pitched downward
    12: 80, 13: 80,    # Front knees flexed forward to ground
    14: 80, 15: 80,    # Rear knees flexed forward to ground
}

#: Default initial pose for the robot and simulation backend.
DEFAULT_POSE: dict[int, int] = STAND_POSE

#: Resting pose (Bittle's `krest` crouch with limbs tucked).
REST_POSE: dict[int, int] = {
    0: 0, 1: 0,
    8: -55, 9: -55, 10: 55, 11: 55,
    12: 60, 13: 60, 14: 60, 15: 60,
}


def resolve(joint: int | str) -> Joint:
    """Look up a joint by index or by name. Raises KeyError if unknown."""
    if isinstance(joint, str):
        if joint.isdigit() or (joint.startswith("-") and joint[1:].isdigit()):
            joint = int(joint)
        else:
            if joint not in BY_NAME:
                raise KeyError(f"unknown joint name: {joint!r}")
            return BY_NAME[joint]
    if joint not in BY_INDEX:
        raise KeyError(f"joint index out of range 0..{DOF - 1}: {joint}")
    return BY_INDEX[joint]
