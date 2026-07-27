"""Backend interface.

A backend is a dumb executor. It receives already-validated commands and does
not re-decide policy -- all policy lives in `app.safety`. Keeping backends
policy-free means adding a transport can't accidentally weaken the safety model.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class CommandResult:
    ok: bool
    sent: str  # the wire bytes, repr'd for logging/UI
    response: str | None = None
    detail: str | None = None
    meta: dict = field(default_factory=dict)


class Backend(abc.ABC):
    """Executes validated commands against a target (sim or real hardware)."""

    name: str = "base"
    is_real_hardware: bool = False

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    @abc.abstractmethod
    async def move_joints(
        self, pairs: list[tuple[int, int]], *, simultaneous: bool = True
    ) -> CommandResult: ...

    @abc.abstractmethod
    async def run_skill(self, skill_name: str) -> CommandResult: ...

    @abc.abstractmethod
    async def rest(self) -> CommandResult:
        """Release servo torque (`d`). Used by the E-stop path."""

    @abc.abstractmethod
    async def read_joints(self) -> dict[int, int]: ...

    @abc.abstractmethod
    async def status(self) -> dict: ...
