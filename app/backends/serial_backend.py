"""Real hardware over USB serial.

Reachable only when BITTLE_ALLOW_REAL_HARDWARE=true AND the request carries the
confirm token -- see app/config.py and app/main.py. Nothing here re-checks
policy; by the time a call lands, safety has already run.

Concurrency: OpenCat has no command IDs. `printSerialMessage` matches a response
by comparing it to the token that was just sent, so two overlapping writers will
happily consume each other's ACKs and report success for the wrong command. A
single lock serialises everything -- correctness over throughput, on a robot.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .. import protocol
from .base import Backend, CommandResult

logger = logging.getLogger(__name__)

#: The firmware echoes the token it accepted. We wait this long for it before
#: treating the command as unacknowledged. ardSerial escalates 3->5->7s; we use a
#: flat, shorter budget because an HTTP caller is waiting on the other end.
ACK_TIMEOUT_SEC = 3.0


class SerialUnavailable(RuntimeError):
    pass


class SerialBackend(Backend):
    name = "serial"
    is_real_hardware = True

    def __init__(self, port: str, baud: int = protocol.BAUD_RATE, *, boot_wait: float = 2.0):
        self.port = port
        self.baud = baud
        self.boot_wait = boot_wait
        self._serial = None
        self._lock = asyncio.Lock()
        self._last_error: str | None = None
        self._connected_at: float | None = None

    async def connect(self) -> None:
        try:
            import serial  # pyserial, imported lazily so sim-only deploys need no device
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SerialUnavailable("pyserial is not installed") from exc

        def _open():
            return serial.Serial(self.port, self.baud, timeout=ACK_TIMEOUT_SEC)

        try:
            self._serial = await asyncio.to_thread(_open)
        except Exception as exc:
            self._last_error = str(exc)
            raise SerialUnavailable(f"cannot open {self.port}: {exc}") from exc

        # The NyBoard resets when the port opens (DTR); commands sent during the
        # bootloader window are simply lost. Waiting is not optional.
        await asyncio.sleep(self.boot_wait)
        if self._serial.in_waiting:
            await asyncio.to_thread(self._serial.reset_input_buffer)
        self._connected_at = time.time()
        logger.info("serial connected: %s @ %s", self.port, self.baud)

    async def close(self) -> None:
        if self._serial is not None:
            await asyncio.to_thread(self._serial.close)
            self._serial = None
            self._connected_at = None

    def _require_port(self):
        if self._serial is None or not self._serial.is_open:
            raise SerialUnavailable(f"serial port {self.port} is not open")
        return self._serial

    async def _write(self, wire: bytes, expect_token: str) -> CommandResult:
        """Write bytes and wait for the firmware's token echo."""
        async with self._lock:
            port = self._require_port()

            def _txn() -> str | None:
                port.reset_input_buffer()
                # ardSerial chunks at 20 bytes with a 1ms gap; the NyBoard's
                # serial buffer is small and drops bytes on long single writes.
                for start in range(0, len(wire), 20):
                    port.write(wire[start : start + 20])
                    port.flush()
                    time.sleep(0.001)
                deadline = time.monotonic() + ACK_TIMEOUT_SEC
                lines: list[str] = []
                while time.monotonic() < deadline:
                    raw = port.readline()
                    if not raw:
                        continue
                    line = raw.decode("ISO-8859-1").strip()
                    if not line:
                        continue
                    lines.append(line)
                    if line.lower().startswith(expect_token.lower()):
                        return "\n".join(lines)
                return "\n".join(lines) or None

            try:
                response = await asyncio.to_thread(_txn)
            except Exception as exc:
                self._last_error = str(exc)
                raise

        acked = bool(response and response.lower().startswith(expect_token.lower()))
        return CommandResult(
            ok=acked,
            sent=wire.decode("latin-1"),
            response=response,
            detail=None if acked else "no matching token echo within timeout",
            meta={"acked": acked},
        )

    async def move_joints(
        self, pairs: list[tuple[int, int]], *, simultaneous: bool = True
    ) -> CommandResult:
        wire = protocol.encode_joint_move(pairs, simultaneous=simultaneous)
        token = (
            protocol.Token.INDEXED_SIMULTANEOUS.value
            if simultaneous
            else protocol.Token.INDEXED_SEQUENTIAL.value
        )
        return await self._write(wire, token)

    async def run_skill(self, skill_name: str) -> CommandResult:
        return await self._write(protocol.encode_skill(skill_name), protocol.Token.SKILL.value)

    async def rest(self) -> CommandResult:
        return await self._write(protocol.encode_rest(), protocol.Token.REST.value)

    async def read_joints(self) -> dict[int, int]:
        result = await self._write(protocol.encode_query_joints(), protocol.Token.JOINTS.value)
        return _parse_joint_response(result.response)

    async def status(self) -> dict:
        return {
            "backend": self.name,
            "real_hardware": True,
            "connected": self._serial is not None and self._serial.is_open,
            "port": self.port,
            "baud": self.baud,
            "connected_at": self._connected_at,
            "last_error": self._last_error,
        }


def _parse_joint_response(response: str | None) -> dict[int, int]:
    """Parse the `j` response into {index: angle}.

    Firmware formatting varies across builds, so this is intentionally lenient:
    it pulls integers out of the payload and positionally maps them. Returns {}
    rather than guessing when the shape is unrecognised -- a wrong joint reading
    is worse than a missing one.
    """
    if not response:
        return {}
    numbers: list[int] = []
    for line in response.splitlines():
        cleaned = line.strip()
        if cleaned.lower().startswith("j"):
            cleaned = cleaned[1:]
        for chunk in cleaned.replace(",", " ").split():
            try:
                numbers.append(int(float(chunk)))
            except ValueError:
                continue
    if not numbers:
        return {}
    return {index: value for index, value in enumerate(numbers[:16])}
