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
from .motion import TrajectoryValidator, get_lifecycle, get_composer, CompositionError
from .motion.builtin_library import BUILTIN_MOVESETS, EXPRESSIVE_MACROS, get_builtin_moveset
from .motion.primitives import JOINT_GROUPS
from .motion.schema import SkillIR
from .profiles import get_registry
from .safety import SafetyError

logger = logging.getLogger("bittle-agent.agent")

SYSTEM_PROMPT = """You are the autonomous movement planner, choreographer, and motion harness for a Petoi Bittle quadruped robot dog.
You control the robot exclusively through tool calls. You do not touch hardware directly.

CORE SAFETY & MOTION RULES:
1. Safety is absolute. Joint angles and mechanical envelopes are strictly enforced.
2. Standard Bittle has 9 servos:
   - Head pan: Joint 0 (yaw: -60° right to +60° left).
   - Front shoulders: Joint 8 (FL), Joint 9 (FR) (pitch forward/back: -110° to 65°).
   - Rear shoulders: Joint 10 (BR), Joint 11 (BL) (pitch forward/back: -110° to 65°).
   - Front knees: Joint 12 (FL), Joint 13 (FR) (pitch: -65° to 110°).
   - Rear knees: Joint 14 (BR), Joint 15 (BL) (pitch: -65° to 110°).
   - Tail: Joint 1 is UNINSTALLED on standard Bittle. Do not command joint 1.
3. Reference Postures:
   - Stand / Neutral balance: {0: 0, 8: -45, 9: -45, 10: -45, 11: -45, 12: 80, 13: 80, 14: 80, 15: 80}
   - Seated: {0: 0, 8: -30, 9: -30, 10: 80, 11: 80, 12: 40, 13: 40, 14: 75, 15: 75}
   - Rest / Belly: {0: 0, 8: -55, 9: -55, 10: 55, 11: 55, 12: 60, 13: 60, 14: 60, 15: 60}
4. Generating Novel Movements & Choreography:
   - You CAN and SHOULD invent your own original movements, poses, and behaviors whenever requested!
   - For single custom poses (e.g. "tilt head right and raise paw"): call `bittle_move_joints` or `bittle_execute_sequence`.
   - For multi-step custom choreography (e.g. "stalk like a cat", "dance", "stealth crouch"): call `bittle_execute_sequence` with an array of keyframe steps (move, pause). Each step specifies angles, delays (e.g. 150-350ms), and interpolation speeds.
   - For persistent new movesets that the user wants to save: call `bittle_save_moveset` with a unique name, description, and list of keyframes.
5. Pre-Defined Skills:
   - For simple requests matching built-in skills ('sit', 'balance', 'hi', 'pu', 'rest', 'bf', 'wkF'): call `bittle_do_skill`.
6. Locomotion Gaits:
   - Gaits ('wkF', 'trF', etc.) move the robot physically across a surface. Only execute if locomotion is acknowledged.
7. Emergency Stop:
   - If unexpected resistance occurs or user requests stop, immediately call `bittle_estop`.
8. Explain intent concisely before acting.

MOTION PRIMITIVE COMPOSITION (preferred for novel moves):
    Instead of specifying every joint angle from scratch, compose from reusable per-limb "primitives".
    Workflow:
    1. `bittle_list_primitives` → see available building blocks (head_scan_left, fr_paw_wave, torso_lower, etc.)
    2. `bittle_compose_move` → snap primitives together (sequential, parallel, or blend mode)
    3. `bittle_evaluate_move` → check safety score & smoothness metrics
    4. `bittle_execute_sequence` → run the composed moveset on the simulator
    5. `bittle_iterate_move` → tweak delays, speeds, angles based on what you observed
    6. `bittle_save_moveset` → persist winners to the library for reuse
    Parallel mode merges joint-disjoint primitives into simultaneous frames (e.g. head_scan + rear_wiggle).
    Sequential mode concatenates primitives end-to-end with optional transition blending.

OBSTACLE COURSES & TERRAIN ADAPTATION:
    You can inspect, spawn, and conquer 3D obstacle challenges in the digital twin:
    1. `bittle_load_course` → spawn a terrain preset:
       - 'mini_stairs': 3-step staircase with 18mm risers (requires high-lift knee flexion >= 95°).
       - 'ramp_bridge': 18° incline ramp and elevated narrow balance plank.
       - 'crawl_tunnel': 65mm ceiling (requires belly crawl `low_tunnel_crawl` or `crF` with body height < 60mm).
       - 'agility_slalom': 4 slalom cones (requires coordinated yaw turns and weave gaits).
    2. `bittle_get_course_layout` → inspect obstacle dimensions, bounds, and physical constraints.
    3. `bittle_evaluate_terrain_clearance` → verify if your candidate moveset kinematically clears the obstacles before execution.
    4. Built-in obstacle gaits: 'stair_step_up', 'ramp_climb', 'low_tunnel_crawl', 'slalom_weave_left', 'slalom_weave_right'.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bittle_execute_sequence",
            "description": "Execute a composite multi-step behavior or motion sequence in one atomic call without multi-turn latency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Descriptive name for the sequence (e.g. 'sit_and_look_around')"},
                    "steps": {
                        "type": "array",
                        "description": "Ordered steps: skill, move (angles map), or pause",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["skill", "move", "pause"]},
                                "skill": {"type": "string", "description": "Skill name e.g. 'sit', 'ck', 'bf', 'hi'"},
                                "angles": {"type": "object", "description": "Joint angles map e.g. {\"0\": 30}"},
                                "delay_ms": {"type": "integer", "description": "Delay in ms after step"},
                                "ack_locomotion": {"type": "boolean", "description": "Required if gait is used"}
                            },
                            "required": ["type"]
                        }
                    },
                    "target": {"type": "string", "enum": ["sim", "real"], "description": "Target backend"}
                },
                "required": ["steps"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_express",
            "description": "Trigger an expressive emotive macro behavior (e.g. 'look_around', 'nod_yes', 'shake_no', 'curious_tilt', 'stretch_and_rest', 'happy_wiggle', 'bow').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "enum": ["look_around", "nod_yes", "shake_no", "curious_tilt", "stretch_and_rest", "happy_wiggle", "bow"],
                        "description": "Expression macro name"
                    },
                    "target": {"type": "string", "enum": ["sim", "real"]}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_save_moveset",
            "description": "Save a named moveset with keyframe angles and delays into the persistent library for 3D timeline replay and reuse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique moveset identifier name"},
                    "description": {"type": "string", "description": "Human-readable description"},
                    "frames": {
                        "type": "array",
                        "description": "Keyframe list with angles and delay_ms",
                        "items": {
                            "type": "object",
                            "properties": {
                                "angles": {"type": "object", "description": "Joint angles map"},
                                "delay_ms": {"type": "integer", "description": "Hold duration in ms"},
                                "speed_deg_per_step": {"type": "integer", "description": "Interpolation speed"}
                            },
                            "required": ["angles"]
                        }
                    }
                },
                "required": ["name", "frames"]
            }
        }
    },
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
    # ── Motion Primitive Composition Tools ────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "bittle_list_primitives",
            "description": "List available motion primitives — reusable per-limb clips that can be composed into novel moves. Each primitive targets a joint group (head, front_left, front_right, rear_left, rear_right, front_legs, rear_legs, torso, all).",
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {
                        "type": "string",
                        "description": "Filter by joint group (e.g. 'head', 'front_left', 'rear_legs', 'torso', 'all'). Omit to list all.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_compose_move",
            "description": "Compose a new moveset by snapping named primitives together. Returns frames, quality metrics, and a preview-ready moveset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique name for the composed moveset"},
                    "primitives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of primitive names to compose",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "blend"],
                        "description": "Composition mode. sequential=end-to-end, parallel=merge joint-disjoint into simultaneous frames, blend=sequential with interpolation frames",
                    },
                    "description": {"type": "string", "description": "Human-readable description"},
                },
                "required": ["name", "primitives"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_iterate_move",
            "description": "Tweak an existing moveset's parameters — scale delays, speeds, offset joint angles, swap primitives, insert/remove frames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of moveset to iterate on"},
                    "adjustments": {
                        "type": "object",
                        "description": "Adjustment parameters: delay_scale (float), speed_scale (float), angle_offsets ({joint:offset}), swap_primitive ({old,new}), insert_frame ({index,frame}), remove_frame (int)",
                    },
                },
                "required": ["name", "adjustments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_evaluate_move",
            "description": "Dry-run validate and score a moveset without executing. Returns smoothness, max delta, duration, and reversal count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Moveset name to evaluate from library"},
                    "frames": {
                        "type": "array",
                        "description": "Or provide raw frames to evaluate inline",
                        "items": {"type": "object"},
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_list_library",
            "description": "List all saved movesets in the library (built-in + user-created + GLM-composed).",
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
            "name": "bittle_load_moveset",
            "description": "Load a previously saved moveset by name for replay, iteration, or inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Moveset name to load"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_load_course",
            "description": "Load a 3D obstacle course preset in the digital twin simulator (mini_stairs, ramp_bridge, crawl_tunnel, agility_slalom, or none).",
            "parameters": {
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["none", "mini_stairs", "ramp_bridge", "crawl_tunnel", "agility_slalom"],
                        "description": "Obstacle course preset name",
                    }
                },
                "required": ["preset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_get_course_layout",
            "description": "Get the active obstacle course layout, obstacle dimensions, and clearance constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "description": "Optional preset name (defaults to active course)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bittle_evaluate_terrain_clearance",
            "description": "Kinematically evaluate whether a candidate moveset or joint posture clears the obstacle course.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_preset": {
                        "type": "string",
                        "enum": ["mini_stairs", "ramp_bridge", "crawl_tunnel", "agility_slalom"],
                        "description": "Target course to evaluate against",
                    },
                    "angles": {
                        "type": "object",
                        "description": "Joint angles map for single pose clearance",
                    },
                    "sequence": {
                        "type": "array",
                        "description": "Array of keyframe step objects",
                    },
                    "moveset_name": {
                        "type": "string",
                        "description": "Name of saved or built-in moveset to evaluate",
                    },
                },
                "required": ["course_preset"],
            },
        },
    },
]


class GLMAgentHarness:
    def __init__(self, controller: Controller, settings: Settings):
        self.controller = controller
        self.settings = settings
        self.lifecycle = get_lifecycle()
        self.composer = get_composer()
        self.active_course = "none"

    def get_system_prompt(self, target: str = "sim") -> str:
        """Construct context-rich system prompt with pre-injected active hardware state and latency directives."""
        prof = self.controller.profile
        installed = sorted(list(prof.installed_joints))
        skills_summary = "sit, balance (stand), rest, up, str, zero, ck (check around), hi (wave hello), pu (pushups), nd (nod), bf (backflip), pee, rc (recover), wkF (walk forward)"
        estop_status = "ENGAGED" if self.controller.safety.estop_engaged else "disengaged"

        return f"""{SYSTEM_PROMPT}

CURRENT ROBOT RUNTIME STATE:
- Target Backend: {target}
- Active Hardware Profile: {prof.profile_id} (Model: {prof.robot_model}, Board: {prof.board})
- Installed Joints: {installed} (Head pan: 0, Front shoulders: 8-9, Rear shoulders: 10-11, Knees: 12-15).
- Joint Envelopes: Head [-60, 60], Shoulders [-110, 65], Knees [-65, 110]. Joint 1 is UNINSTALLED.
- Common Pre-Approved Skills: {skills_summary}
- E-Stop: {estop_status} | Locomotion Allowed: {self.settings.allow_locomotion}

REAL-TIME LATENCY DIRECTIVE:
Capabilities, joint envelopes, and robot status are already pre-loaded into your context above.
DO NOT waste turns calling `bittle_list_capabilities`, `bittle_get_hardware_profile`, or `bittle_status` unless the operator specifically asks for diagnostics.
- Predefined single actions: call `bittle_do_skill` immediately on Turn 1.
- Novel poses or custom choreography: call `bittle_execute_sequence` with your chosen keyframe angles and delays on Turn 1.
- Creating / saving persistent movesets: call `bittle_save_moveset` on Turn 1.
- Expressive emotive gestures: call `bittle_express`.
Execute user movement goals immediately on Turn 1."""

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
                    "moveset": get_builtin_moveset(resolved.name),
                }

            if name == "bittle_execute_sequence":
                seq_name = args.get("name", "custom_sequence")
                steps = args.get("steps", [])
                prof = self.controller.profile
                tier = "agent" if target == "real" else "transport"

                # 1. Atomic pre-flight dry run: validate all steps before execution
                for idx, step in enumerate(steps):
                    stype = step.get("type")
                    if stype == "skill":
                        sname = step.get("skill", "")
                        resolved_skill = skills_mod.resolve(sname)
                        if resolved_skill.locomotes and not step.get("ack_locomotion", False):
                            raise SafetyError(
                                reason="locomotion_unacked",
                                detail=f"Step {idx} ({sname}) locomotes across floor and requires ack_locomotion=true"
                            )
                    elif stype == "move":
                        angles = step.get("angles", {})
                        for k in angles:
                            j = joints_mod.resolve(k)
                            if not prof.is_installed(j.index):
                                raise SafetyError(
                                    reason="unused_joint",
                                    detail=f"Joint {k} is uninstalled on profile {prof.profile_id}"
                                )

                # 2. Execution of validated steps
                executed_steps = []
                moveset_frames = []
                total_duration = 0

                for step in steps:
                    stype = step.get("type")
                    delay = int(step.get("delay_ms", 150))
                    total_duration += delay

                    if stype == "skill":
                        sname = step["skill"]
                        ack = step.get("ack_locomotion", False)
                        res, resolved = await self.controller.skill(
                            sname, target=target, ack_locomotion=ack, confirm=confirm
                        )
                        executed_steps.append({"type": "skill", "skill": resolved.name, "ok": res.ok})
                        b_mset = get_builtin_moveset(resolved.name)
                        if b_mset:
                            moveset_frames.extend(b_mset["frames"])
                        else:
                            moveset_frames.append({"angles": {}, "label": resolved.name, "delay_ms": delay})

                    elif stype == "move":
                        angles_in = step["angles"]
                        res, val = await self.controller.move(
                            angles_in, target=target, confirm=confirm, envelope_tier=tier
                        )
                        applied_map = {int(i): a for i, a in val.pairs}
                        executed_steps.append({"type": "move", "angles": applied_map, "ok": res.ok})
                        moveset_frames.append({
                            "angles": applied_map,
                            "delay_ms": delay,
                            "speed_deg_per_step": int(step.get("speed_deg_per_step", 8))
                        })

                    elif stype == "pause":
                        executed_steps.append({"type": "pause", "delay_ms": delay})
                        if moveset_frames:
                            moveset_frames[-1]["delay_ms"] += delay

                moveset = {
                    "name": seq_name,
                    "frames": moveset_frames,
                    "total_duration_ms": total_duration,
                }
                return {
                    "ok": True,
                    "name": seq_name,
                    "executed_steps": executed_steps,
                    "moveset": moveset,
                }

            if name == "bittle_express":
                expr = args.get("expression", "")
                macro = EXPRESSIVE_MACROS.get(expr)
                if not macro:
                    return {
                        "ok": False,
                        "error": f"unknown expression: {expr}. Available: {list(EXPRESSIVE_MACROS.keys())}"
                    }
                res = await self.execute_tool("bittle_execute_sequence", {
                    "name": expr,
                    "steps": macro,
                    "target": target,
                }, target_override=target, confirm_token=confirm)
                if isinstance(res, dict):
                    res["expression"] = expr
                return res

            if name == "bittle_save_moveset":
                m_name = args["name"]
                desc = args.get("description", "")
                frames = args["frames"]
                payload = {
                    "name": m_name,
                    "description": desc,
                    "frames": frames,
                    "kind": "custom",
                }
                saved = self.lifecycle.save_moveset(m_name, payload)
                get_composer().save_moveset(m_name, payload)
                return {"ok": True, "name": m_name, "moveset": saved}

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

            # ── Motion Primitive Composition Tools ────────────────────
            if name == "bittle_list_primitives":
                group = args.get("group")
                prims = self.composer.list_primitives(group=group)
                return {
                    "ok": True,
                    "primitives": [
                        {
                            "name": p["name"],
                            "group": p["group"],
                            "joints": p["joints"],
                            "description": p.get("description", ""),
                            "tags": p.get("tags", []),
                            "frame_count": len(p.get("frames", [])),
                        }
                        for p in prims
                    ],
                    "joint_groups": JOINT_GROUPS,
                }

            if name == "bittle_compose_move":
                m_name = args["name"]
                prim_names = args["primitives"]
                mode = args.get("mode", "sequential")
                desc = args.get("description", "")
                moveset = self.composer.compose_move(
                    m_name, prim_names, mode=mode, description=desc
                )
                return {"ok": True, "moveset": moveset}

            if name == "bittle_iterate_move":
                m_name = args["name"]
                adjustments = args.get("adjustments", {})
                moveset = self.composer.iterate_move(m_name, adjustments)
                return {"ok": True, "moveset": moveset}

            if name == "bittle_evaluate_move":
                m_name = args.get("name")
                frames = args.get("frames")
                metrics = self.composer.evaluate_move(name=m_name, frames=frames)
                return {"ok": True, "metrics": metrics}

            if name == "bittle_list_library":
                movesets = self.composer.list_movesets()
                return {
                    "ok": True,
                    "movesets": [
                        {
                            "name": m.get("name", ""),
                            "description": m.get("description", ""),
                            "frame_count": len(m.get("frames", [])),
                            "source_primitives": m.get("source_primitives", []),
                            "created_at": m.get("created_at", ""),
                        }
                        for m in movesets
                    ],
                }

            if name == "bittle_load_moveset":
                m_name = args["name"]
                moveset = self.composer.get_moveset(m_name)
                if moveset is None:
                    return {"ok": False, "error": f"Moveset {m_name!r} not found"}
                return {"ok": True, "moveset": moveset}

            # ── Obstacle Course & Terrain Tools ─────────────────────────
            if name == "bittle_load_course":
                preset = args["preset"]
                from .motion.obstacles import COURSE_LAYOUTS, get_course_layout
                if preset not in COURSE_LAYOUTS:
                    return {"ok": False, "error": f"Unknown preset '{preset}'. Choose from {list(COURSE_LAYOUTS.keys())}"}
                self.active_course = preset
                layout = get_course_layout(preset)
                return {
                    "ok": True,
                    "preset": preset,
                    "description": layout.get("description", ""),
                    "layout": layout,
                }

            if name == "bittle_get_course_layout":
                preset = args.get("preset") or self.active_course
                from .motion.obstacles import get_course_layout
                layout = get_course_layout(preset)
                return {"ok": True, "preset": preset, "layout": layout}

            if name == "bittle_evaluate_terrain_clearance":
                from .motion.obstacles import evaluate_terrain_clearance
                c_preset = args["course_preset"]
                data_to_eval = None
                if "sequence" in args:
                    data_to_eval = args["sequence"]
                elif "angles" in args:
                    data_to_eval = args["angles"]
                elif "moveset_name" in args:
                    m_name = args["moveset_name"]
                    m = get_builtin_moveset(m_name) or self.composer.get_moveset(m_name)
                    if m:
                        data_to_eval = m.get("frames", [])
                    else:
                        return {"ok": False, "error": f"Moveset '{m_name}' not found in library."}
                else:
                    return {"ok": False, "error": "Provide either 'angles', 'sequence', or 'moveset_name' to evaluate."}

                return evaluate_terrain_clearance(c_preset, data_to_eval)

            return {"ok": False, "error": f"unknown tool: {name}"}

        except CompositionError as exc:
            logger.warning("Composition rejected tool %s: %s", name, str(exc))
            return {"ok": False, "error": "composition_error", "detail": str(exc)}
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
        """Run multi-turn agent loop communicating with local GLM instance via real-time SSE streaming."""
        api_base, model = await self.resolve_endpoint_and_model()
        api_key = self.settings.llm_api_key or "EMPTY"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        full_messages = [{"role": "system", "content": self.get_system_prompt(target=target)}]
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
                    "stream": True,
                }
                url = f"{api_base}/chat/completions"

                thought_acc = ""
                content_acc = ""
                tool_calls_acc: dict[int, dict[str, Any]] = {}

                try:
                    async with client.stream("POST", url, headers=headers, json=payload) as stream_resp:
                        if stream_resp.status_code >= 400:
                            err_body = await stream_resp.aread()
                            err_text = err_body.decode("utf-8", errors="replace")
                            yield {
                                "type": "error",
                                "error": f"http_{stream_resp.status_code}",
                                "detail": err_text,
                                "content": f"Model endpoint error {stream_resp.status_code}: {err_text}",
                            }
                            return

                        async for line in stream_resp.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            raw_data = line[5:].strip()
                            if raw_data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(raw_data)
                            except Exception:
                                continue

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            # 1. Reasoning / thought tokens from GLM
                            reasoning_chunk = delta.get("reasoning")
                            if reasoning_chunk:
                                thought_acc += reasoning_chunk
                                yield {"type": "thought", "content": reasoning_chunk}

                            # 2. Text response tokens
                            content_chunk = delta.get("content")
                            if content_chunk:
                                content_acc += content_chunk
                                yield {"type": "text", "content": content_chunk}

                            # 3. Incremental tool calls delta
                            delta_tcs = delta.get("tool_calls")
                            if delta_tcs:
                                for dtc in delta_tcs:
                                    idx = dtc.get("index", 0)
                                    if idx not in tool_calls_acc:
                                        tool_calls_acc[idx] = {
                                            "id": dtc.get("id") or f"call_{idx}",
                                            "name": "",
                                            "arguments": "",
                                        }
                                    if dtc.get("id"):
                                        tool_calls_acc[idx]["id"] = dtc["id"]
                                    fn = dtc.get("function", {})
                                    if fn.get("name"):
                                        tool_calls_acc[idx]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        tool_calls_acc[idx]["arguments"] += fn["arguments"]

                except Exception as exc:
                    yield {
                        "type": "error",
                        "error": "endpoint_error",
                        "detail": str(exc),
                        "content": f"GLM endpoint error ({api_base} / {model}): {exc}",
                    }
                    return

                # If no tool calls were requested, yield completion and finish
                if not tool_calls_acc:
                    yield {"type": "done", "final_message": content_acc}
                    return

                # Build assistant message with accumulated tool calls
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if content_acc:
                    assistant_msg["content"] = content_acc
                assistant_msg["tool_calls"] = [
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        },
                    }
                    for _, item in sorted(tool_calls_acc.items())
                ]
                full_messages.append(assistant_msg)

                # Execute all requested tool calls
                for _, item in sorted(tool_calls_acc.items()):
                    fn_name = item["name"]
                    tc_id = item["id"]
                    try:
                        args = json.loads(item["arguments"]) if item["arguments"] else {}
                    except Exception:
                        args = {}

                    yield {
                        "type": "tool_call",
                        "name": fn_name,
                        "args": args,
                        "id": tc_id,
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
                        "id": tc_id,
                    }

                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_res),
                    })
