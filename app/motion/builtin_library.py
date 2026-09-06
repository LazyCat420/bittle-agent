"""Authentic keyframe sequences for Bittle skills and expressive macros.

Provides multi-frame trajectory definitions with per-joint angles in degrees,
speeds, and delays for both the 3D digital twin and the hardware execution pipeline.
"""

from __future__ import annotations
from typing import Any

# Standard Bittle joint indices:
# 0: Head pan
# 8: Shoulder FL, 9: Shoulder FR, 10: Shoulder BR, 11: Shoulder BL
# 12: Knee FL,     13: Knee FR,     14: Knee BR,     15: Knee BL

STAND_ANGLES = {0: 0, 8: -45, 9: -45, 10: -45, 11: -45, 12: 80, 13: 80, 14: 80, 15: 80}
SIT_ANGLES = {0: 0, 8: -30, 9: -30, 10: 80, 11: 80, 12: 40, 13: 40, 14: 75, 15: 75}
REST_ANGLES = {0: 0, 8: -55, 9: -55, 10: 55, 11: 55, 12: 60, 13: 60, 14: 60, 15: 60}

BUILTIN_MOVESETS: dict[str, dict[str, Any]] = {
    "sit": {
        "name": "sit",
        "label": "Sit",
        "description": "Standard Bittle seated posture",
        "kind": "posture",
        "frames": [
            {"angles": SIT_ANGLES, "speed_deg_per_step": 8, "delay_ms": 300}
        ]
    },
    "balance": {
        "name": "balance",
        "label": "Stand / Balance",
        "description": "Neutral four-legged standing posture",
        "kind": "posture",
        "frames": [
            {"angles": STAND_ANGLES, "speed_deg_per_step": 8, "delay_ms": 300}
        ]
    },
    "rest": {
        "name": "rest",
        "label": "Rest (Relax)",
        "description": "Flat rested posture on belly",
        "kind": "posture",
        "frames": [
            {"angles": REST_ANGLES, "speed_deg_per_step": 8, "delay_ms": 300}
        ]
    },
    "ck": {
        "name": "ck",
        "label": "Check Around",
        "description": "Scans left and right with head pan while standing steady",
        "kind": "behavior",
        "frames": [
            {"angles": {**STAND_ANGLES, 0: 45}, "speed_deg_per_step": 6, "delay_ms": 350},
            {"angles": {**STAND_ANGLES, 0: -45}, "speed_deg_per_step": 6, "delay_ms": 350},
            {"angles": {**STAND_ANGLES, 0: 30}, "speed_deg_per_step": 6, "delay_ms": 250},
            {"angles": {**STAND_ANGLES, 0: 0}, "speed_deg_per_step": 6, "delay_ms": 200},
        ]
    },
    "hi": {
        "name": "hi",
        "label": "Wave Hello",
        "description": "Sits down and waves front-right paw twice",
        "kind": "behavior",
        "frames": [
            {"angles": SIT_ANGLES, "speed_deg_per_step": 8, "delay_ms": 250},
            {"angles": {**SIT_ANGLES, 9: 20, 13: -25}, "speed_deg_per_step": 8, "delay_ms": 200},
            {"angles": {**SIT_ANGLES, 9: 35, 13: 15}, "speed_deg_per_step": 12, "delay_ms": 180},
            {"angles": {**SIT_ANGLES, 9: 15, 13: -30}, "speed_deg_per_step": 12, "delay_ms": 180},
            {"angles": {**SIT_ANGLES, 9: 35, 13: 15}, "speed_deg_per_step": 12, "delay_ms": 180},
            {"angles": {**SIT_ANGLES, 9: 20, 13: -25}, "speed_deg_per_step": 8, "delay_ms": 200},
            {"angles": SIT_ANGLES, "speed_deg_per_step": 8, "delay_ms": 250},
        ]
    },
    "pu": {
        "name": "pu",
        "label": "Push Ups",
        "description": "Lowers front chest to floor and presses up two times",
        "kind": "behavior",
        "frames": [
            {"angles": {**STAND_ANGLES, 8: -75, 9: -75, 12: 105, 13: 105}, "speed_deg_per_step": 6, "delay_ms": 350},
            {"angles": STAND_ANGLES, "speed_deg_per_step": 8, "delay_ms": 300},
            {"angles": {**STAND_ANGLES, 8: -75, 9: -75, 12: 105, 13: 105}, "speed_deg_per_step": 6, "delay_ms": 350},
            {"angles": STAND_ANGLES, "speed_deg_per_step": 8, "delay_ms": 300},
        ]
    },
    "nd": {
        "name": "nd",
        "label": "Nod",
        "description": "Affirmative nodding motion",
        "kind": "behavior",
        "frames": [
            {"angles": {**STAND_ANGLES, 8: -55, 9: -55, 10: -35, 11: -35}, "speed_deg_per_step": 8, "delay_ms": 200},
            {"angles": {**STAND_ANGLES, 8: -35, 9: -35, 10: -55, 11: -55}, "speed_deg_per_step": 8, "delay_ms": 200},
            {"angles": {**STAND_ANGLES, 8: -55, 9: -55, 10: -35, 11: -35}, "speed_deg_per_step": 8, "delay_ms": 200},
            {"angles": STAND_ANGLES, "speed_deg_per_step": 8, "delay_ms": 200},
        ]
    },
    "bf": {
        "name": "bf",
        "label": "Backflip",
        "description": "High-energy backward somersault landing squarely on all four feet",
        "kind": "behavior",
        "frames": [
            # 1. Deep crouch
            {"angles": {0: 0, 8: -80, 9: -80, 10: 75, 11: 75, 12: 110, 13: 110, 14: 70, 15: 70}, "speed_deg_per_step": 12, "delay_ms": 250},
            # 2. Explosive spring extension
            {"angles": {0: 0, 8: 30, 9: 30, 10: -90, 11: -90, 12: 20, 13: 20, 14: 120, 15: 120}, "speed_deg_per_step": 30, "delay_ms": 150},
            # 3. Tuck in air
            {"angles": {0: 0, 8: -70, 9: -70, 10: 70, 11: 70, 12: 90, 13: 90, 14: 90, 15: 90}, "speed_deg_per_step": 30, "delay_ms": 200},
            # 4. Impact landing absorption
            {"angles": {0: 0, 8: -60, 9: -60, 10: -60, 11: -60, 12: 95, 13: 95, 14: 95, 15: 95}, "speed_deg_per_step": 16, "delay_ms": 180},
            # 5. Recover to standard stand
            {"angles": STAND_ANGLES, "speed_deg_per_step": 8, "delay_ms": 250},
        ]
    },
    "pee": {
        "name": "pee",
        "label": "Pee",
        "description": "Playful single rear-leg lift posture",
        "kind": "behavior",
        "frames": [
            {"angles": {**STAND_ANGLES, 10: -30, 14: 40}, "speed_deg_per_step": 6, "delay_ms": 800},
            {"angles": STAND_ANGLES, "speed_deg_per_step": 6, "delay_ms": 300},
        ]
    },
    "str": {
        "name": "str",
        "label": "Stretch",
        "description": "Full body waking stretch",
        "kind": "behavior",
        "frames": [
            {"angles": {**STAND_ANGLES, 8: -75, 9: -75, 10: -20, 11: -20, 12: 110, 13: 110, 14: 50, 15: 50}, "speed_deg_per_step": 5, "delay_ms": 600},
            {"angles": STAND_ANGLES, "speed_deg_per_step": 6, "delay_ms": 300},
        ]
    },
}

EXPRESSIVE_MACROS: dict[str, list[dict[str, Any]]] = {
    "look_around": [
        {"type": "move", "angles": {0: 40}, "delay_ms": 250},
        {"type": "move", "angles": {0: -40}, "delay_ms": 250},
        {"type": "move", "angles": {0: 20}, "delay_ms": 200},
        {"type": "move", "angles": {0: 0}, "delay_ms": 150},
    ],
    "nod_yes": [
        {"type": "move", "angles": {8: -55, 9: -55, 10: -35, 11: -35}, "delay_ms": 180},
        {"type": "move", "angles": {8: -35, 9: -35, 10: -55, 11: -55}, "delay_ms": 180},
        {"type": "move", "angles": {8: -55, 9: -55, 10: -35, 11: -35}, "delay_ms": 180},
        {"type": "move", "angles": STAND_ANGLES, "delay_ms": 150},
    ],
    "shake_no": [
        {"type": "move", "angles": {0: 30}, "delay_ms": 150},
        {"type": "move", "angles": {0: -30}, "delay_ms": 150},
        {"type": "move", "angles": {0: 25}, "delay_ms": 120},
        {"type": "move", "angles": {0: -25}, "delay_ms": 120},
        {"type": "move", "angles": {0: 0}, "delay_ms": 150},
    ],
    "curious_tilt": [
        {"type": "move", "angles": {0: 25, 8: -50, 9: -40}, "delay_ms": 400},
        {"type": "move", "angles": {0: -25, 8: -40, 9: -50}, "delay_ms": 400},
        {"type": "move", "angles": STAND_ANGLES, "delay_ms": 200},
    ],
    "stretch_and_rest": [
        {"type": "skill", "skill": "str"},
        {"type": "pause", "delay_ms": 400},
        {"type": "skill", "skill": "rest"},
    ],
    "happy_wiggle": [
        {"type": "move", "angles": {10: -30, 11: -60, 14: 65, 15: 95}, "delay_ms": 120},
        {"type": "move", "angles": {10: -60, 11: -30, 14: 95, 15: 65}, "delay_ms": 120},
        {"type": "move", "angles": {10: -30, 11: -60, 14: 65, 15: 95}, "delay_ms": 120},
        {"type": "move", "angles": {10: -60, 11: -30, 14: 95, 15: 65}, "delay_ms": 120},
        {"type": "move", "angles": STAND_ANGLES, "delay_ms": 150},
    ],
    "bow": [
        {"type": "move", "angles": {8: -70, 9: -70, 12: 100, 13: 100, 10: -35, 11: -35}, "delay_ms": 500},
        {"type": "move", "angles": STAND_ANGLES, "delay_ms": 250},
    ],
}


def get_builtin_moveset(name: str) -> dict[str, Any] | None:
    return BUILTIN_MOVESETS.get(name)


def list_builtin_movesets() -> list[dict[str, Any]]:
    return list(BUILTIN_MOVESETS.values())
