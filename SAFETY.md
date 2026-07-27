# SAFETY

What protects the hardware, what doesn't, and where the sharp edges are.

## Sources

Everything below is transcribed from the firmware and reference client, not
inferred. If you change a constant, re-check it against these:

| Fact | Source |
|---|---|
| Token list (`k`, `m`, `i`, `j`, `d`, `G`, …) | `PetoiCamp/OpenCat` `src/OpenCat.h` |
| Bittle `angleLimit` / `middleShift` tables | `src/OpenCat.h`, `#elif defined BITTLE` |
| Skill names (`sitI` → `ksit`) | `src/InstinctBittle.h` |
| ASCII vs binary encoding, 115200 baud, `~`/`\n` | `serialMaster/ardSerial.py` |
| `DOF 16`, `WALKING_DOF 8` | `src/OpenCat.h` |

## The one that will bite you

**Firmware-legal angles are not always wire-representable.**

Bittle's knees are `{-80, 200}` and shoulders `{-200, 80}`. Binary tokens
(`I`, `L`, `M`, `K`) pack angles with `struct.pack('b', …)` — signed char,
**−128..127**.

So `200` is a legal angle the wire cannot carry. Encoded naively it becomes
**−56**, which is not a garbled no-op: it drives the joint hard toward the
*opposite* limit. A validator that clamps only to the firmware table and then
sends a binary token is more dangerous than no validator, because it looks
correct.

Two independent mitigations:

1. Clamp to `firmware_limit ∩ (−128, 127)` — `Joint.safe_min` / `safe_max`.
2. `encode_binary()` **raises** rather than wrapping, so a future caller that
   bypasses the clamp still cannot silently emit a wrapped angle.

Joint moves use the ASCII `i`/`m` tokens, which have no such ceiling. The
clamp is therefore conservative for ASCII — the honest trade is that both
encodings obey one limit table, rather than a limit that changes with transport.
Joints affected are reported with `wire_clipped: true` on `/api/joints`.

## Guarantees

| Guard | Behaviour | Test |
|---|---|---|
| Sim by default | `target` defaults to `sim`; `real` needs env flag **and** confirm token | `test_real_target_is_refused_when_hardware_disabled` |
| Angle clamp | per-joint, asymmetric, from the firmware table | `test_firmware_legal_angle_beyond_wire_range_is_clamped_not_wrapped` |
| No wrap | binary encoder raises out of range | `test_binary_encoder_refuses_out_of_range_rather_than_wrapping` |
| Unknown joint / skill | rejected with a structured reason | `test_unknown_joint_index_is_rejected` |
| Unpopulated slots | joints 4–7 refused | `test_unused_joint_slot_is_rejected` |
| Non-finite angles | NaN/inf rejected before the clamp | `test_nan_angle_is_rejected_and_does_not_slip_through_the_clamp` |
| Locomotion | gaits need `ack_locomotion` | `test_gait_requires_explicit_ack` |
| Rate limit | token bucket bounds sustained duty cycle | `test_sustained_flooding_is_rate_limited` |
| E-stop | latching; blocks motion with 409 | `test_estop_blocks_subsequent_motion_with_409` |
| E-stop priority | works when rate limit is exhausted | `test_estop_works_even_when_rate_limit_is_exhausted` |

NaN deserves its own note: it fails *every* `<` comparison, so `max(lo, min(hi, nan))`
returns `nan` and passes a clamp unchanged. It is rejected explicitly, before the
clamp, rather than being trusted to fall out of the arithmetic.

## Deliberate non-guarantees

Stating these plainly is part of the safety model — a guard people *think* exists
is worse than one they know doesn't.

- **E-stop makes the robot go limp.** `d` (`T_REST`) releases servo torque, so a
  standing robot drops into a crouch and could fall from a height. This is the
  right trade against a stalling servo, but it is not a graceful stop.
- **No balance or collision protection.** The service validates *commands*, not
  *consequences*. A legal pose can still topple the robot; a legal gait can still
  walk it off a table. Hence the locomotion ack.
- **The sim is kinematic, not physical.** It models joint state, limits and
  timing — not contact, balance or falling. A pose that "works in sim" is only
  evidence that the command was well-formed.
- **No servo current/temperature feedback.** The rate limiter is an open-loop
  proxy for duty cycle. It cannot detect a stalled or overheating servo.
- **Joint readback is best-effort.** `j` response formatting varies across
  firmware builds; the parser returns `{}` rather than guessing, so absence of
  readings is not evidence the robot is at the commanded pose.
- **Serial has no command IDs.** Responses are matched by echoed token, so a
  single lock serialises all traffic. Bypassing that lock would silently
  mis-attribute ACKs.

## Enabling real hardware

Two independent switches, both required:

```bash
BITTLE_ALLOW_REAL_HARDWARE=true      # deployment-level arm
BITTLE_CONFIRM_TOKEN=<secret>        # per-request confirmation
BITTLE_SERIAL_PORT=/dev/ttyUSB0
```

Leaving `BITTLE_CONFIRM_TOKEN` empty while hardware is armed means *no token is
required*; the service logs a warning at startup rather than pretending it is
safe. To disable gaits entirely, set `BITTLE_ALLOW_LOCOMOTION=false`.

The container needs the device passed through (`devices:` in
`docker-compose.yml`, commented out by default) — a sim-only deployment must not
have a serial device mapped in at all.

## If something goes wrong

1. Hit **E-STOP** in the UI, or `POST /api/estop`. It is never rate-limited.
2. Physically power off the robot — the definitive stop. Software E-stop only
   helps if the service and link are healthy.
3. `POST /api/estop/clear` to resume. There is intentionally no MCP tool for
   this: an agent may stop the robot, only a human may restart it.
