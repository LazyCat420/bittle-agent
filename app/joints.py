"""Bittle joint map and angle limits.

Every constant here is transcribed from the OpenCat firmware, not inferred:
  - angleLimit / middleShift : src/OpenCat.h, `#elif defined BITTLE` block
  - DOF = 16, WALKING_DOF = 8 : src/OpenCat.h

Do not "tidy" these numbers. The table is deliberately asymmetric because the
hardware is asymmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

DOF = 16
WALKING_DOF = 8

# Binary tokens (I/L/M/K) pack angles with struct.pack('b') -> signed char.
# Anything outside this range cannot be represented on the wire at all.
WIRE_MIN = -128
WIRE_MAX = 127


@dataclass(frozen=True)
class Joint:
    index: int
    name: str
    label: str
    fw_min: int  # firmware angleLimit lower bound
    fw_max: int  # firmware angleLimit upper bound
    used: bool = True

    @property
    def safe_min(self) -> int:
        """Firmware limit intersected with what the wire can actually carry.

        The intersection matters: firmware permits knee angles up to 200, but a
        binary token would wrap 200 to -56 (signed char). Clamping to the
        firmware table ALONE is the bug this property exists to prevent.
        """
        return max(self.fw_min, WIRE_MIN)

    @property
    def safe_max(self) -> int:
        return min(self.fw_max, WIRE_MAX)

    @property
    def wire_clipped(self) -> bool:
        """True when the wire encoding, not the hardware, is the binding limit."""
        return self.fw_min < WIRE_MIN or self.fw_max > WIRE_MAX


# Index layout follows OpenCat's "ordered by distance from torso" convention:
#   0-3   head / tail group   (Bittle uses 0 = head pan, 1 = tail)
#   4-7   unused on Bittle (no shoulder-roll servos; slots exist for Nybble parity)
#   8-11  shoulders  (upper leg)
#   12-15 knees      (lower leg)
#
# Leg ordering across 8-15 is front-left, right-front, right-back, left-back,
# matching middleShift {55, 55, -55, -55} and rotationDirection {1, -1, -1, 1}.
_JOINTS: tuple[Joint, ...] = (
    Joint(0, "head", "Head pan", -120, 120),
    Joint(1, "tail", "Tail", -85, 85),
    Joint(2, "reserved_2", "Reserved", -120, 120, used=False),
    Joint(3, "reserved_3", "Reserved", -120, 120, used=False),
    Joint(4, "unused_4", "Unused", -90, 60, used=False),
    Joint(5, "unused_5", "Unused", -90, 60, used=False),
    Joint(6, "unused_6", "Unused", -90, 90, used=False),
    Joint(7, "unused_7", "Unused", -90, 90, used=False),
    Joint(8, "shoulder_front_left", "Shoulder front-left", -200, 80),
    Joint(9, "shoulder_front_right", "Shoulder front-right", -200, 80),
    Joint(10, "shoulder_back_right", "Shoulder back-right", -80, 200),
    Joint(11, "shoulder_back_left", "Shoulder back-left", -80, 200),
    Joint(12, "knee_front_left", "Knee front-left", -80, 200),
    Joint(13, "knee_front_right", "Knee front-right", -80, 200),
    Joint(14, "knee_back_right", "Knee back-right", -80, 200),
    Joint(15, "knee_back_left", "Knee back-left", -80, 200),
)

BY_INDEX: dict[int, Joint] = {j.index: j for j in _JOINTS}
BY_NAME: dict[str, Joint] = {j.name: j for j in _JOINTS}

#: Joints an operator or LLM may command. Excludes the unused Nybble-parity slots
#: so a stray index can't drive a servo that isn't physically there.
CONTROLLABLE: tuple[Joint, ...] = tuple(j for j in _JOINTS if j.used)
CONTROLLABLE_INDICES: frozenset[int] = frozenset(j.index for j in CONTROLLABLE)

#: Authentic standing posture where all 4 legs support the robot upright.
#: Derived from MuJoCo keyframe 'stand' and CAD FK ground alignment:
#: Thighs pitched down -45°, knees flexed forward to ground +80°.
#: All 4 feet land within 0.1 mm of Z = -53.2 mm.
STAND_POSE: dict[int, int] = {
    0: 0,   # Head pan centered
    1: 0,   # Tail centered
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
