"""OpenCat serial token encoding.

Mirrors `serialMaster/ardSerial.py::serialWriteNumToByte`:

  lowercase token -> ASCII, space-separated args, '\\n' terminated
  uppercase token -> struct.pack('b', ...) binary args, '~' terminated

The two encodings are NOT interchangeable, and the difference is a safety
boundary rather than a style choice -- see `encode_binary`.
"""

from __future__ import annotations

import struct
from enum import StrEnum

BAUD_RATE = 115200
ASCII_TERMINATOR = b"\n"
BINARY_TERMINATOR = b"~"


class Token(StrEnum):
    """Tokens from OpenCat.h. Only the ones we deliberately support."""

    SKILL = "k"                    # T_SKILL          -- ksit, kwkF ...
    INDEXED_SEQUENTIAL = "m"       # T_INDEXED_SEQUENTIAL_ASC   (one joint after another)
    INDEXED_SIMULTANEOUS = "i"     # T_INDEXED_SIMULTANEOUS_ASC (all joints together)
    JOINTS = "j"                   # T_JOINTS  -- "j" all angles, "j 8" one joint
    REST = "d"                     # T_REST    -- releases servo torque
    GYRO_BALANCE = "G"             # T_GYRO_BALANCE (toggle)
    PRINT_GYRO = "v"               # T_PRINT_GYRO
    QUERY = "?"                    # T_QUERY   -- model/version handshake
    PAUSE = "p"                    # T_PAUSE


class EncodingError(ValueError):
    """Raised when a value cannot be represented on the wire."""


def encode_ascii(token: str, args: list[int] | None = None) -> bytes:
    """Encode a lowercase/ASCII token.

    Preferred for joint motion: ASCII carries arbitrary integers, so the
    signed-char ceiling that constrains binary tokens does not apply.
    """
    payload = ""
    for value in args or []:
        payload += f"{round(value)} "
    return token.encode() + payload.encode("utf-8") + ASCII_TERMINATOR


def encode_binary(token: str, args: list[int] | None = None) -> bytes:
    """Encode an uppercase/binary token.

    Args are packed as SIGNED CHAR. Values outside -128..127 are not
    representable. We raise rather than let struct wrap them, because a wrapped
    angle is not a garbled no-op -- 200 becomes -56 and drives the joint hard
    toward the opposite limit. Silent wrapping here would break a servo.
    """
    values = [round(v) for v in (args or [])]
    for value in values:
        if not -128 <= value <= 127:
            raise EncodingError(
                f"value {value} exceeds signed-char range for binary token "
                f"{token!r}; use the ASCII token instead"
            )
    return token.encode() + struct.pack("b" * len(values), *values) + BINARY_TERMINATOR


def encode_joint_move(pairs: list[tuple[int, int]], simultaneous: bool = True) -> bytes:
    """Encode a joint move as `i idx ang idx ang ...` (or `m` for sequential).

    Defaults to simultaneous. Sequential motion visits intermediate poses that
    the caller never asked for -- with several joints queued, a leg can swing
    through a collision on its way. Simultaneous is the safer default.
    """
    args: list[int] = []
    for index, angle in pairs:
        args.extend((index, angle))
    token = Token.INDEXED_SIMULTANEOUS if simultaneous else Token.INDEXED_SEQUENTIAL
    return encode_ascii(token.value, args)


def encode_skill(skill_name: str) -> bytes:
    """Encode `ksit`-style skill invocation. `skill_name` excludes the 'k'."""
    return encode_ascii(f"{Token.SKILL.value}{skill_name}")


def encode_rest() -> bytes:
    """`d` -- rest posture, releasing servo torque. Used by the E-stop."""
    return encode_ascii(Token.REST.value)


def encode_query_joints(index: int | None = None) -> bytes:
    return encode_ascii(Token.JOINTS.value, [index] if index is not None else None)
