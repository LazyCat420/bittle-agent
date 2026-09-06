"""Local GLM Agent Harness for Petoi Bittle.

Connects to an internal, local OpenAI-compatible endpoint (e.g. GLM-4 / GLM-4-9B
hosted on Gold Spark / DGX Spark or Jetson via vLLM, Ollama, or LM Studio).
The model never touches serial directly; all actions pass through the Controller,
HardwareProfile, and SafetyValidator.
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
from .motion import TrajectoryValidator, get_lifecycle
from .motion.schema import SkillIR
from .profiles import get_registry
from .safety import SafetyError

logger = logging.getLogger("bittle-agent.agent")

SYSTEM_PROMPT = """You are the autonomous movement planner and motion harness for a Petoi Bittle quadruped robot dog.
You control the robot exclusively through tool calls. You do not touch hardware directly.

CORE SAFETY RULES:
1. Safety is absolute. Joint angles and mechanical envelopes are strictly enforced.
2. Standard Bittle has 9 servos (Head pan 0, Front shoulders 8-9, Rear shoulders 10-11, Knees 12-15). Do not command uninstalled joints.
3. Locomotion gaits ('wkF', 'trF', etc.) move the robot physically across a surface. Only execute if locomotion is acknowledged.
4. For predefined behaviors and postures, call `bittle_do_skill` with verified skill names ('sit', 'balance', 'hi', 'pu', 'rest').
5. For complex custom multi-frame motions, follow the safe authoring lifecycle:
   a. `bittle_draft_skill`: specify typed frames, speeds, and delays.
   b. `bittle_validate_skill`: verify delta limits, reversal budgets, and agent envelopes.
   c. `bittle_simulate_skill`: test in simulation before requesting physical promotion.
   d. `bittle_run_approved_skill`: execute promoted skills with cryptographic SHA-256 verification.
6. If unexpected resistance occurs or user requests stop, immediately call `bittle_estop`.
7. Explain intent concisely before acting.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bittle_list_capabilities",
            "description": "List what the Bittle robot can do: verified skills (postures, gaits, behaviors) and controllable joints with their safe angle limits.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_get_hardware_profile",
            "description": "Get the active verified hardware profile (robot model, board, installed joints, feedback capability, and profile hash).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_get_joint_envelopes",
            "description": "Get the 4-tier joint angle envelopes (firmware, transport, tested mechanical clearance, and agent operational limits).",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
            "description": "Set specific joint angles in degrees. On real hardware, angles are clamped to the conservative agent operational envelope.",
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
            "name": "bittle_draft_skill",
            "description": "Draft a multi-frame custom skill/behavior without executing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "object",
                        "description": "Skill IR definition containing name, profile_id, kind, loop_count, and frames array.",
                    }
                },
                "required": ["skill"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_validate_skill",
            "description": "Deterministically validate a skill IR against delta bounds, reversal chatter limits, cumulative travel budgets, and agent envelopes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "object", "description": "Skill IR definition to validate."}
                },
                "required": ["skill"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_simulate_skill",
            "description": "Simulate a validated skill in the digital twin simulator and record verification telemetry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest_hash": {"type": "string", "description": "Payload hash of the validated skill."}
                },
                "required": ["manifest_hash"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_run_approved_skill",
            "description": "Execute a promoted, immutable skill manifest by hash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest_hash": {"type": "string", "description": "SHA-256 payload hash of approved skill."},
                    "target": {"type": "string", "enum": ["sim", "real"], "description": "Target backend."},
                },
                "required": ["manifest_hash"],
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
        self.lifecycle = get_lifecycle()

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

            if name == "bittle_get_hardware_profile":
                prof = self.controller.profile
                return {
                    "ok": True,
                    "profile_id": prof.profile_id,
                    "robot_model": prof.robot_model,
                    "board": prof.board,
                    "servo_model": prof.servo_model,
                    "feedback_capable": prof.feedback_capable,
                    "installed_joints": list(prof.installed_joints),
                    "profile_hash": prof.profile_hash(),
                }

            if name == "bittle_get_joint_envelopes":
                prof = self.controller.profile
                return {
                    "ok": True,
                    "profile_id": prof.profile_id,
                    "envelopes": {
                        j.index: {
                            "name": j.name,
                            "firmware": [j.fw_min, j.fw_max],
                            "transport": [j.transport_min, j.transport_max],
                            "tested": [j.tested_min, j.tested_max],
                            "agent": [j.agent_min, j.agent_max],
                            "installed": prof.is_installed(j.index),
                        }
                        for j in joints_mod.CONTROLLABLE
                    },
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
                # LLM agent moves on real hardware must be held strictly within agent operational envelope
                tier = "agent" if target == "real" else "transport"
                result, validated = await self.controller.move(
                    angles_in,
                    target=target,
                    simultaneous=simultaneous,
                    confirm=confirm,
                    envelope_tier=tier,
                )
                return {
                    "ok": result.ok,
                    "sent": result.sent.strip(),
                    "response": result.response,
                    "envelope_tier": tier,
                    "clamped": validated.clamped,
                    "adjustments": [a.__dict__ for a in validated.adjustments],
                    "applied": {str(i): a for i, a in validated.pairs},
                }

            if name == "bittle_draft_skill":
                raw_skill = args["skill"]
                if "profile_id" not in raw_skill:
                    raw_skill["profile_id"] = self.controller.profile.profile_id
                ir = self.lifecycle.draft(raw_skill)
                return {"ok": True, "skill_ir": ir.model_dump(mode="json")}

            if name == "bittle_validate_skill":
                raw_skill = args["skill"]
                if "profile_id" not in raw_skill:
                    raw_skill["profile_id"] = self.controller.profile.profile_id
                ir = SkillIR.model_validate(raw_skill)
                val_res, manifest = self.lifecycle.validate_and_compile(ir)
                return {
                    "ok": val_res.valid,
                    "errors": val_res.errors,
                    "warnings": val_res.warnings,
                    "budget": val_res.budget.model_dump() if val_res.budget else None,
                    "manifest_hash": manifest.payload_hash if manifest else None,
                }

            if name == "bittle_simulate_skill":
                m_hash = args["manifest_hash"]
                manifest = self.lifecycle.get_manifest(m_hash)
                # Simulate trajectory on SimBackend
                res, _ = await self.controller.run_approved_skill(m_hash, target="sim")
                evidence = {"status": "success", "frames_simulated": len(manifest.ir.frames)}
                self.lifecycle.record_simulation(m_hash, evidence)
                return {
                    "ok": True,
                    "manifest_hash": m_hash,
                    "status": "simulated",
                    "frames": len(manifest.ir.frames),
                }

            if name == "bittle_run_approved_skill":
                m_hash = args["manifest_hash"]
                res, manifest = await self.controller.run_approved_skill(
                    m_hash, target=target, confirm=confirm
                )
                return {
                    "ok": res.ok,
                    "manifest_hash": m_hash,
                    "skill_name": manifest.ir.name,
                    "sent": res.sent,
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
    async def resolve_endpoint_and_model(self) -> tuple[str, str]:
        """Resolve the active LLM endpoint and model, querying cluster candidates if needed."""
        primary_base = self.settings.llm_api_base.rstrip("/")
        candidates = [
            primary_base,
            "http://10.0.0.141:8000/v1",
            "http://10.0.0.16:5591/vllm-shim/gold-spark/v1",
            "http://10.0.0.30:8000/v1",
        ]
        unique_candidates: list[str] = []
        for c in candidates:
            c_norm = c.rstrip("/")
            if c_norm not in unique_candidates:
                unique_candidates.append(c_norm)

        api_key = self.settings.llm_api_key or "EMPTY"
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=2.5) as client:
            for base in unique_candidates:
                try:
                    res = await client.get(f"{base}/models", headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                        if self.settings.llm_model in models:
                            return base, self.settings.llm_model
                        if models:
                            return base, models[0]
                        return base, self.settings.llm_model
                except Exception:
                    continue

        return primary_base, self.settings.llm_model

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        target: str = "sim",
        confirm_token: str | None = None,
        max_turns: int = 6,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run multi-turn agent loop communicating with local GLM instance."""
        api_base, model = await self.resolve_endpoint_and_model()
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
                    res.raise_for_status()
                    data = res.json()
                except Exception as exc:
                    yield {"type": "error", "content": f"GLM endpoint error ({api_base} / {model}): {exc}"}
                    return

                choice = data["choices"][0]
                message = choice["message"]
                full_messages.append(message)

                if message.get("content"):
                    yield {"type": "text", "content": message["content"]}

                tool_calls = message.get("tool_calls", [])
                if not tool_calls:
                    return

                for tc in tool_calls:
                    fn = tc["function"]
                    fn_name = fn["name"]
                    try:
                        args = json.loads(fn["arguments"])
                    except Exception:
                        args = {}

                    yield {
                        "type": "tool_call",
                        "name": fn_name,
                        "args": args,
                        "id": tc.get("id"),
                    }

                    tool_res = await self.execute_tool(
                        fn_name,
                        args,
                        target_override=target,
                        confirm_token=confirm_token,
                    )

                    yield {
                        "type": "tool_result",
                        "name": fn_name,
                        "result": tool_res,
                        "id": tc.get("id"),
                    }

                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(tool_res),
                    })
