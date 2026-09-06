"""Local GLM Agent Harness for Petoi Bittle.

Connects to an internal, local OpenAI-compatible endpoint (e.g. GLM-4 / GLM-4-9B
hosted on Gold Spark / DGX Spark or Jetson via vLLM, Ollama, or LM Studio).
The model never touches serial directly; all actions pass through the Controller
and SafetyValidator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx

from . import joints as joints_mod
from . import skills as skills_mod
from .config import Settings
from .controller import Controller, TargetUnavailable
from .safety import SafetyError

logger = logging.getLogger("bittle-agent.agent")

SYSTEM_PROMPT = """You are the autonomous movement planner and motion harness for a Petoi Bittle quadruped robot dog.
You control the robot exclusively through tool calls. You do not touch hardware directly.

CORE RULES:
1. Safety is absolute. Joint angles are hardware-constrained. If a joint clamp is reported back to you, respect it.
2. Locomotion gaits (such as 'wkF' walk forward, 'trF' trot forward, 'bk' back up) move the robot physically across a surface. Only execute gaits if locomotion has been acknowledged or requested.
3. For predefined behaviors and postures, prefer calling `bittle_do_skill` with verified skill names (e.g. 'sit', 'balance', 'hi', 'pu', 'snf', 'pee', 'rest').
4. For custom poses, use `bittle_move_joints` with joint names or OpenCat indices.
5. If something goes wrong, unexpected resistance occurs, or the user commands stop, immediately call `bittle_estop`.
6. Explain your intent concisely before or as you act. When you finish an action sequence, summarize what the robot performed.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bittle_list_capabilities",
            "description": "List what the Bittle robot can do: verified skills (postures, gaits, behaviors) and controllable joints with their safe angle limits.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_status",
            "description": "Report current robot status: active target, E-stop state, rate limit headroom, and current joint angles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["sim", "real"],
                        "description": "Target backend ('sim' default or 'real').",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_do_skill",
            "description": "Run a named Bittle skill/instinct (posture, gait, or behavior).",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Skill name (e.g. 'sit', 'balance', 'hi', 'wkF', 'trF', 'pu').",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["sim", "real"],
                        "description": "Target backend ('sim' default or 'real').",
                    },
                    "ack_locomotion": {
                        "type": "boolean",
                        "description": "Must be true for gaits that move across the floor.",
                    },
                },
                "required": ["skill"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_move_joints",
            "description": "Set specific joint angles in degrees. Angles outside safe limits are clamped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angles": {
                        "type": "object",
                        "description": "Map of joint name or index to angle in degrees, e.g. {\"head\": 20, \"knee_front_left\": 45}.",
                        "additionalProperties": {"type": "number"},
                    },
                    "target": {
                        "type": "string",
                        "enum": ["sim", "real"],
                        "description": "Target backend ('sim' or 'real').",
                    },
                    "simultaneous": {
                        "type": "boolean",
                        "description": "Move all joints simultaneously (default true, safer).",
                    },
                },
                "required": ["angles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_estop",
            "description": "EMERGENCY STOP. Immediately releases servo torque and latches motion block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for triggering E-stop.",
                    }
                },
                "required": [],
            },
        },
    },
]


class GLMAgentHarness:
    def __init__(self, controller: Controller, settings: Settings):
        self.controller = controller
        self.settings = settings

    async def execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        target_override: str | None = None,
        confirm_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute an agent tool call against the controller safely."""
        target = target_override or args.get("target", "sim")
        confirm = confirm_token or self.settings.confirm_token

        try:
            if name == "bittle_list_capabilities":
                return {
                    "ok": True,
                    "skills": [
                        {"name": s.name, "label": s.label, "kind": s.kind, "locomotes": s.locomotes}
                        for s in skills_mod.ALL
                    ],
                    "joints": self.controller.describe_joints(),
                    "note": "Angle min/max are effective safe limits.",
                }

            if name == "bittle_status":
                status = await self.controller.status(target)
                angles = await self.controller.joint_state(target, confirm)
                return {"ok": True, "status": status, "angles": angles}

            if name == "bittle_do_skill":
                skill_name = args["skill"]
                ack_locomotion = args.get("ack_locomotion", False)
                result, resolved = await self.controller.skill(
                    skill_name,
                    target=target,
                    ack_locomotion=ack_locomotion,
                    confirm=confirm,
                )
                return {
                    "ok": result.ok,
                    "skill": resolved.name,
                    "token": resolved.token,
                    "sent": result.sent.strip(),
                    "response": result.response,
                    "meta": result.meta,
                }

            if name == "bittle_move_joints":
                angles_in = args["angles"]
                simultaneous = args.get("simultaneous", True)
                result, validated = await self.controller.move(
                    angles_in,
                    target=target,
                    simultaneous=simultaneous,
                    confirm=confirm,
                )
                return {
                    "ok": result.ok,
                    "sent": result.sent.strip(),
                    "response": result.response,
                    "clamped": validated.clamped,
                    "adjustments": [a.__dict__ for a in validated.adjustments],
                    "applied": {str(i): a for i, a in validated.pairs},
                }

            if name == "bittle_estop":
                reason = args.get("reason", "agent_invoked")
                res = await self.controller.estop(reason)
                return {"ok": True, "estop": "engaged", "reason": reason, "backends": res["backends"]}

            return {"ok": False, "error": f"unknown tool: {name}"}

        except SafetyError as exc:
            logger.warning("Safety gate rejected tool %s: %s (%s)", name, exc.reason, exc.detail)
            return {"ok": False, "error": "safety_refused", "reason": exc.reason, "detail": exc.detail}
        except TargetUnavailable as exc:
            return {"ok": False, "error": "target_unavailable", "detail": str(exc)}
        except Exception as exc:
            logger.exception("Tool %s failed: %s", name, exc)
            return {"ok": False, "error": "execution_failed", "detail": str(exc)}

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        target: str = "sim",
        confirm_token: str | None = None,
        max_turns: int = 6,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run multi-turn agent loop communicating with local GLM instance."""
        api_base = self.settings.llm_api_base.rstrip("/")
        model = self.settings.llm_model
        api_key = self.settings.llm_api_key or "EMPTY"
        url = f"{api_base}/chat/completions"

        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages:
            full_messages.append(m)

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            for turn in range(max_turns):
                payload = {
                    "model": model,
                    "messages": full_messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "temperature": 0.2,
                }

                try:
                    res = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                except httpx.RequestError as exc:
                    yield {
                        "type": "error",
                        "error": "local_llm_unreachable",
                        "detail": f"Could not connect to local model endpoint at {url}: {exc}",
                    }
                    return

                if res.status_code != 200:
                    yield {
                        "type": "error",
                        "error": "llm_error",
                        "status": res.status_code,
                        "detail": res.text[:500],
                    }
                    return

                data = res.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []

                if content:
                    yield {"type": "thought", "content": content}

                if not tool_calls:
                    full_messages.append({"role": "assistant", "content": content})
                    yield {"type": "done", "final_message": content}
                    return

                full_messages.append(msg)

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    tc_id = tc.get("id", "call_default")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}

                    yield {"type": "tool_call", "name": name, "args": args, "id": tc_id}

                    result = await self.execute_tool(
                        name,
                        args,
                        target_override=target,
                        confirm_token=confirm_token,
                    )

                    yield {"type": "tool_result", "name": name, "result": result, "id": tc_id}

                    full_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )

            yield {
                "type": "done",
                "final_message": "Action sequence completed (max turns reached).",
            }
