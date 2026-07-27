# bittle-agent — Plan

An LLM-controllable Petoi Bittle, where **the model never touches the serial port**.

```
LLM  ──MCP tool call──▶  lazy-tool-service  ──HTTP──▶  bittle-agent API
                                                            │
                                                    ┌───────▼────────┐
                                                    │ SAFETY VALIDATOR│  ← mandatory, not bypassable
                                                    └───────┬────────┘
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                                    SimBackend (default)            SerialBackend (opt-in)
                                    kinematic model                 /dev/ttyUSB0 @115200
```

## Why this shape

The interesting failure mode is not "the LLM sends a bad word", it's "the LLM sends a
*plausible* number that the firmware accepts and the servo tears itself apart". So the
validator is a real translation layer with knowledge of the hardware, not a regex.

Everything below is grounded in the actual OpenCat firmware and `ardSerial.py`, not
guessed. Sources are cited in `SAFETY.md`.

---

## §1 Protocol facts that shape the design

These were verified against `PetoiCamp/OpenCat` `src/OpenCat.h`, `src/InstinctBittle.h`
and `serialMaster/ardSerial.py`. Each one changes the code.

**1. Two wire encodings, not one.** Lowercase tokens are ASCII, space-separated,
`\n`-terminated. Uppercase tokens are binary `struct.pack('b', ...)`, `~`-terminated.
`ardSerial.py:70-140`.

**2. Binary angles are signed char.** `'b'` packing means the wire can only carry
**−128..127**.

**3. …but the firmware's own limits exceed that.** Bittle's `angleLimit` table has
knees at `{-80, 200}` and shoulders at `{-200, 80}`. So an angle that is *legal per
firmware* (e.g. 200) is *unsendable as binary* — `struct.pack('b', 200)` raises, and a
naive `& 0xFF` would wrap it to −56 and drive the joint to the opposite extreme.

> This is the single most important finding. A validator that clamps only to the
> firmware table and then sends binary is **actively dangerous**. We clamp to the
> intersection of (firmware limit ∩ wire-representable range), and we prefer the ASCII
> `i`/`m` tokens, which have no such limit.

**4. Per-joint limits are asymmetric.** There is no global ±125. Index 1 (tail) is
`{-85, 85}`, index 0 (head) is `{-120, 120}`, knees are `{-80, 200}`. A single global
clamp is wrong in both directions — too loose for the tail, too tight for the knees.

**5. Real skill names come from `InstinctBittle.h`.** Skills are stored with an `I`/`N`
suffix which is *dropped* and prefixed with `k`: `sitI` → `ksit`, `wkFI` → `kwkF`. We
ship the verified list and reject anything not on it, because an unknown `k` token is
silently ignored by firmware — the robot just keeps doing whatever it was doing, which
looks like a hang.

**6. `m` is sequential, `i` is simultaneous.** `m0 70 8 -20` moves joints one after
another; `i0 70 8 -20` moves them together. For multi-joint poses, sequential motion
passes through configurations that simultaneous motion never visits — a leg can collide
mid-sequence. Multi-joint writes therefore default to `i`.

**7. The robot is not transactional.** `printSerialMessage` reads with escalating 3→5→7s
timeouts and matches the echoed token. There is no command ID. Two concurrent writers
interleave and mis-attribute each other's ACKs, so the serial backend holds a single
lock and is strictly one-command-at-a-time.

## §2 Safety model

Ordered by how much they actually protect the hardware:

| Guard | Rule |
|---|---|
| Default target | `sim`. Reaching real hardware takes an explicit env flag **and** a per-request confirm token. |
| Angle clamp | per-joint `firmware_limit ∩ wire_range`, from the real table |
| Unknown joint | reject; index must be in the Bittle DOF map |
| Unknown skill | reject against the verified `InstinctBittle.h` list |
| Rate limit | token bucket; bounds sustained servo duty cycle |
| Serial mutex | one in-flight command, so ACKs can't be mis-attributed |
| E-stop | latching. Once tripped, every motion 409s until explicitly cleared |
| Fail-closed | if the safety layer errors, the command does **not** pass |

E-stop sends `d` (`T_REST`), which releases servo torque. Note this makes the robot go
limp — correct for "stop actuating", but it **will drop to the floor** from a standing
pose. That is the right trade (a burning servo is worse), but it is a real consequence,
documented rather than hidden.

## §3 Scope

**In:** safety validator + tests, sim backend, serial backend, FastAPI, control panel UI,
MCP tools, Docker/deploy.

**Out (deliberate):** PyBullet. The original plan called for it, but a physics sim is a
multi-week project on its own (URDF, contact tuning, gait stability) and none of it makes
the *hardware* safer. We ship a fast kinematic sim that models joint state, limits and
timing — enough to validate command flow end-to-end — and leave a `SimBackend` interface
that PyBullet can implement later. Flagged rather than silently dropped.

**Out:** BLE. Serial is what `ardSerial.py` supports and what the protocol is defined
against; BLE would be a second unverified transport.

## §4 Layout

```
app/
  safety.py       # clamps, whitelists, e-stop, rate limit — the core
  protocol.py     # token encoding; the ASCII/binary split
  joints.py       # verified Bittle joint map + angleLimit table
  skills.py       # verified skill list
  backends/       # sim.py, serial.py, base.py
  main.py         # FastAPI
static/           # control panel
mcp/              # tool schemas for lazy-tool-service
tests/            # safety + encoding tests
```

## §5 Verification

The claim "it is safe" has to be falsifiable, so the tests assert the *dangerous* cases:
angle 200 must not become −56 on the wire; a rejected joint must produce zero writes; a
tripped e-stop must 409; the tail's ±85 must not be widened to ±125.
