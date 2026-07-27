"""Verified Bittle skill list.

Transcribed from PetoiCamp/OpenCat `src/InstinctBittle.h`. Skills are stored with
an `I` (instinct) or `N` (newbility) suffix; the serial API drops the suffix and
prefixes `k` -- `sitI` -> `ksit`, `wkFI` -> `kwkF`.

Why whitelist at all: firmware *silently ignores* an unrecognised `k` token. The
robot keeps executing its previous gait, which presents as "the command hung"
rather than "the command was rejected". Failing loudly here turns a confusing
hardware non-response into a clear 400.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str  # token WITHOUT the leading 'k'
    label: str
    kind: str  # posture | gait | behavior
    locomotes: bool = False

    @property
    def token(self) -> str:
        return f"k{self.name}"


# kind="gait" entries move the robot across the floor -- see `locomotes`, which
# the safety layer uses to require an explicit ack before the robot drives off a
# table. Names are exactly those in InstinctBittle.h minus the suffix.
_SKILLS: tuple[Skill, ...] = (
    # postures
    Skill("balance", "Stand / balance", "posture"),
    Skill("sit", "Sit", "posture"),
    Skill("rest", "Rest (servos relaxed)", "posture"),
    Skill("zero", "Zero / calibration pose", "posture"),
    Skill("up", "Stand up", "posture"),
    Skill("str", "Stretch", "posture"),
    Skill("lnd", "Lifted / landing pose", "posture"),
    Skill("calib", "Calibration posture", "posture"),
    # gaits (locomoting)
    Skill("wkF", "Walk forward", "gait", locomotes=True),
    Skill("wkL", "Walk left", "gait", locomotes=True),
    Skill("trF", "Trot forward", "gait", locomotes=True),
    Skill("trL", "Trot left", "gait", locomotes=True),
    Skill("crF", "Crawl forward", "gait", locomotes=True),
    Skill("crL", "Crawl left", "gait", locomotes=True),
    Skill("vtF", "Step in place forward", "gait", locomotes=True),
    Skill("vtL", "Step in place left", "gait", locomotes=True),
    Skill("bk", "Back up", "gait", locomotes=True),
    Skill("bkL", "Back up left", "gait", locomotes=True),
    Skill("phF", "Push-up walk forward", "gait", locomotes=True),
    Skill("phL", "Push-up walk left", "gait", locomotes=True),
    Skill("bdF", "Bound forward", "gait", locomotes=True),
    Skill("jpF", "Jump forward", "gait", locomotes=True),
    # behaviors
    Skill("hi", "Wave hello", "behavior"),
    Skill("hds", "Handstand", "behavior"),
    Skill("pu", "Push up", "behavior"),
    Skill("chr", "Cheer", "behavior"),
    Skill("ck", "Check around", "behavior"),
    Skill("dg", "Dig", "behavior"),
    Skill("ff", "Fart / play", "behavior"),
    Skill("fiv", "High five", "behavior"),
    Skill("gdb", "Good boy", "behavior"),
    Skill("hg", "Hug", "behavior"),
    Skill("hsk", "Handshake", "behavior"),
    Skill("hu", "Hands up", "behavior"),
    Skill("jmp", "Jump", "behavior"),
    Skill("kc", "Kick", "behavior"),
    Skill("mw", "Meow / sound", "behavior"),
    Skill("nd", "Nod", "behavior"),
    Skill("pd", "Play dead", "behavior"),
    Skill("pee", "Pee", "behavior"),
    Skill("rc", "Recover", "behavior"),
    Skill("rl", "Roll", "behavior"),
    Skill("scrh", "Scratch", "behavior"),
    Skill("snf", "Sniff", "behavior"),
    Skill("tbl", "Table / balance beam", "behavior"),
    Skill("ts", "Test", "behavior"),
    Skill("wh", "Wave hand", "behavior"),
    Skill("ang", "Angry", "behavior"),
    Skill("bf", "Backflip", "behavior"),
    Skill("bx", "Box", "behavior"),
    Skill("cmh", "Come here", "behavior"),
    Skill("zz", "Sleep", "behavior"),
)

BY_NAME: dict[str, Skill] = {s.name: s for s in _SKILLS}
ALL: tuple[Skill, ...] = _SKILLS

#: Skills that carry the robot across a surface -- gated behind an explicit ack.
LOCOMOTING: frozenset[str] = frozenset(s.name for s in _SKILLS if s.locomotes)


def resolve(name: str) -> Skill:
    """Accept either `sit` or `ksit`; return the Skill or raise KeyError."""
    candidate = name.strip()
    if candidate.startswith("k") and candidate not in BY_NAME:
        candidate = candidate[1:]
    if candidate not in BY_NAME:
        raise KeyError(f"unknown skill: {name!r}")
    return BY_NAME[candidate]
