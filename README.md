# bittle-agent

Safety-gated control plane for a [Petoi Bittle](https://www.petoi.com/) quadruped.
An LLM can make the robot move — but **it never touches the serial port**.

```
LLM ──MCP tool──▶ lazy-tool-service ──HTTP──▶ bittle-agent ──▶ SAFETY ──▶ sim | serial
```

Simulation is the default target. Moving a real servo requires a deployment-level
env flag *and* a per-request confirm token *and* the serial device mapped into
the container — three independent switches, so no single mistake reaches the
hardware.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q          # 36 tests
npm run dev                                   # http://localhost:8008
```

Open `http://localhost:8008/` for the control panel. It starts in **SIM**; the
REAL toggle is inert unless hardware is armed.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | liveness |
| `GET` | `/api/status` | E-stop, rate limit, backend state |
| `GET` | `/api/joints` | joint map with **effective** safe ranges |
| `GET` | `/api/joints/state` | last known angles |
| `GET` | `/api/skills` | verified skill list, `locomotes` flagged |
| `POST` | `/api/move` | set joint angles (clamped, adjustments reported) |
| `POST` | `/api/skill` | run a named skill |
| `POST` | `/api/preview` | show the wire bytes **without sending** |
| `POST` | `/api/estop` | latching emergency stop |
| `POST` | `/api/estop/clear` | human-only reset |

Safety refusals return structured errors — `{"error", "reason", "detail"}` with
`400` (invalid), `403` (target blocked), `409` (E-stop), `429` (rate limit) — so
a model can tell "you asked for something unsafe" apart from "the service broke".

```bash
curl -X POST localhost:8008/api/skill -H 'content-type: application/json' \
  -d '{"skill":"sit"}'

# Gaits drive the robot across the floor, so they need an explicit ack:
curl -X POST localhost:8008/api/skill -H 'content-type: application/json' \
  -d '{"skill":"wkF","ack_locomotion":true}'
```

## What the safety layer actually does

Full detail in [SAFETY.md](SAFETY.md). The headline:

> Bittle's firmware permits knee angles up to **200°**, but binary serial tokens
> pack angles as a **signed char** (−128..127). Encoded naively, 200 becomes
> **−56** — driving the joint hard the *opposite* way. Clamping to the firmware
> table alone is worse than useless here, so angles are clamped to
> `firmware_limit ∩ wire_range` and the binary encoder *raises* rather than
> wrapping.

Also enforced: per-joint asymmetric limits (the tail is ±85, not ±125),
whitelisted skills (firmware silently ignores unknown ones, which reads as a
hang), NaN rejection before clamping, a servo-duty-cycle rate limit, a latching
E-stop that outranks the rate limiter, and a single serial lock because OpenCat
has no command IDs.

The tests assert the *dangerous* cases, so the safety claim is falsifiable:

```
test_firmware_legal_angle_beyond_wire_range_is_clamped_not_wrapped
test_nan_angle_is_rejected_and_does_not_slip_through_the_clamp
test_estop_works_even_when_rate_limit_is_exhausted
```

## Enabling real hardware

Read [SAFETY.md](SAFETY.md) first — especially the non-guarantees. Then:

```bash
# .env
BITTLE_ALLOW_REAL_HARDWARE=true
BITTLE_CONFIRM_TOKEN=<secret>
BITTLE_SERIAL_PORT=/dev/ttyUSB0
```

…and uncomment the `devices:` / `group_add:` block in `docker-compose.yml`. A
sim-only deployment should have **no serial device mapped in at all** — absence
of the device is the outermost safety layer.

## MCP tools

`mcp/bittle.json` defines `bittle_list_capabilities`, `bittle_status`,
`bittle_do_skill`, `bittle_move_joints` and `bittle_estop`, matching the shared
tool-registry schema. `mcp/client.py` bridges them to this API.

There is deliberately **no `clear_estop` tool**: an agent may stop the robot,
only a human may restart it.

## Deploy

```bash
npm run deploy            # build + ship to the NAS, port 8008
npm run deploy -- --dry-run
```

`deploy.sh` runs the safety tests as a pre-flight gate and refuses to deploy
robot control code with failing clamps.

## Layout

```
app/
  safety.py     # clamps, whitelists, E-stop, rate limit
  protocol.py   # OpenCat token encoding (ASCII vs binary)
  joints.py     # verified joint map + firmware angleLimit table
  skills.py     # verified skill list
  controller.py # the ONLY path from API to backend
  backends/     # sim.py (default), serial_backend.py
static/         # control panel
mcp/            # tool schemas + bridge client
tests/          # 36 safety/API tests
```

Protocol constants are transcribed from `PetoiCamp/OpenCat`
(`src/OpenCat.h`, `src/InstinctBittle.h`, `serialMaster/ardSerial.py`) — see the
source table in SAFETY.md before changing any of them.
