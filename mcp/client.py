"""Bridge client for lazy-tool-service.

Maps the MCP tool names in bittle.json onto bittle-agent's HTTP API. This is the
only thing the tool layer needs; the model never sees a serial port, a token, or
a raw angle that hasn't been through the validator.

Note there is deliberately NO clear_estop tool. Tripping an emergency stop is
something an agent should be able to do freely; UN-tripping one is a decision
that needs a human who can see the robot.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = os.getenv("BITTLE_AGENT_URL", "http://10.0.0.16:8008")
CONFIRM_TOKEN = os.getenv("BITTLE_CONFIRM_TOKEN", "")
TIMEOUT = float(os.getenv("BITTLE_AGENT_TIMEOUT", "15"))


class BittleClient:
    def __init__(self, base_url: str = BASE_URL, confirm_token: str = CONFIRM_TOKEN):
        self.base_url = base_url.rstrip("/")
        self.confirm_token = confirm_token

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                res = await client.request(method, f"{self.base_url}{path}", **kwargs)
            except httpx.RequestError as exc:
                return {"ok": False, "error": "unreachable", "detail": str(exc)}

        try:
            body = res.json()
        except ValueError:
            return {"ok": False, "error": "bad_response", "detail": res.text[:500]}

        # Pass safety refusals through as structured data rather than raising.
        # The model needs to READ why it was refused so it can correct itself --
        # an exception string would just get retried verbatim.
        if res.status_code >= 400:
            return {
                "ok": False,
                "error": body.get("error", "request_failed"),
                "reason": body.get("reason"),
                "detail": body.get("detail", ""),
                "status": res.status_code,
            }
        return body

    async def list_capabilities(self) -> dict[str, Any]:
        joints = await self._request("GET", "/api/joints")
        skills = await self._request("GET", "/api/skills")
        status = await self._request("GET", "/api/status", params={"target": "sim"})
        return {
            "ok": True,
            "skills": skills.get("skills", []),
            "joints": joints.get("joints", []),
            "real_hardware_allowed": status.get("real_hardware_allowed", False),
            "estop": (status.get("safety") or {}).get("estop"),
            "note": (
                "Angle min/max are effective safe limits. Gaits (locomotes=true) "
                "require ack_locomotion and clear floor space."
            ),
        }

    async def status(self, target: str = "sim") -> dict[str, Any]:
        return await self._request("GET", "/api/status", params={"target": target})

    async def do_skill(
        self, skill: str, target: str = "sim", ack_locomotion: bool = False
    ) -> dict[str, Any]:
        payload = {
            "skill": skill,
            "target": target,
            "ack_locomotion": ack_locomotion,
            "confirm": self.confirm_token or None,
        }
        return await self._request("POST", "/api/skill", json=payload)

    async def move_joints(
        self,
        angles: dict[str, float],
        target: str = "sim",
        simultaneous: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "angles": angles,
            "target": target,
            "simultaneous": simultaneous,
            "confirm": self.confirm_token or None,
        }
        return await self._request("POST", "/api/move", json=payload)

    async def estop(self, reason: str = "agent") -> dict[str, Any]:
        return await self._request("POST", "/api/estop", json={"reason": reason})

    async def execute_sequence(
        self,
        steps: list[dict[str, Any]],
        name: str = "custom_sequence",
        target: str = "sim",
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "steps": steps,
            "target": target,
            "confirm": self.confirm_token or None,
        }
        return await self._request("POST", "/api/sequence", json=payload)

    async def save_moveset(
        self,
        name: str,
        frames: list[dict[str, Any]],
        description: str = "",
        source_primitives: list[str] | None = None,
        composition_mode: str = "",
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "description": description,
            "frames": frames,
            "source_primitives": source_primitives or [],
            "composition_mode": composition_mode,
        }
        return await self._request("POST", "/api/library", json=payload)

    async def list_primitives(self, group: str | None = None) -> dict[str, Any]:
        params = {"group": group} if group else None
        return await self._request("GET", "/api/primitives", params=params)

    async def compose_move(
        self,
        name: str,
        primitives: list[str],
        mode: str = "sequential",
        description: str = "",
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "primitives": primitives,
            "mode": mode,
            "description": description,
        }
        return await self._request("POST", "/api/compose", json=payload)

    async def iterate_move(
        self,
        name: str,
        adjustments: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {"name": name, "adjustments": adjustments}
        return await self._request("POST", "/api/iterate", json=payload)

    async def evaluate_move(
        self,
        name: str | None = None,
        frames: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if name:
            payload["name"] = name
        if frames:
            payload["frames"] = frames
        return await self._request("POST", "/api/evaluate", json=payload)

    async def list_library(self) -> dict[str, Any]:
        return await self._request("GET", "/api/library")

    async def load_moveset(self, name: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/library/{name}")


_client = BittleClient()

#: Dispatch table consumed by lazy-tool-service. Names must match mcp/bittle.json.
TOOLS = {
    "bittle_list_capabilities": lambda **kw: _client.list_capabilities(),
    "bittle_status": lambda target="sim", **kw: _client.status(target),
    "bittle_do_skill": lambda skill, target="sim", ack_locomotion=False, **kw: _client.do_skill(
        skill, target, ack_locomotion
    ),
    "bittle_move_joints": lambda angles, target="sim", simultaneous=True, **kw: _client.move_joints(
        angles, target, simultaneous
    ),
    "bittle_estop": lambda reason="agent", **kw: _client.estop(reason),
    "bittle_execute_sequence": lambda steps, name="custom_sequence", target="sim", **kw: _client.execute_sequence(
        steps, name=name, target=target
    ),
    "bittle_save_moveset": lambda name, frames, description="", source_primitives=None, composition_mode="", **kw: _client.save_moveset(
        name, frames, description=description, source_primitives=source_primitives, composition_mode=composition_mode
    ),
    "bittle_list_primitives": lambda group=None, **kw: _client.list_primitives(group=group),
    "bittle_compose_move": lambda name, primitives, mode="sequential", description="", **kw: _client.compose_move(
        name, primitives, mode=mode, description=description
    ),
    "bittle_iterate_move": lambda name, adjustments, **kw: _client.iterate_move(name, adjustments),
    "bittle_evaluate_move": lambda name=None, frames=None, **kw: _client.evaluate_move(name=name, frames=frames),
    "bittle_list_library": lambda **kw: _client.list_library(),
    "bittle_load_moveset": lambda name, **kw: _client.load_moveset(name),
}


async def execute(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Entry point: execute a bittle_* tool call."""
    handler = TOOLS.get(tool_name)
    if handler is None:
        return {"ok": False, "error": "unknown_tool", "detail": tool_name}
    return await handler(**(arguments or {}))

