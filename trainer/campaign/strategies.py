"""Scripted config-search strategies for the campaign driver (no LLM)."""

from __future__ import annotations

from typing import Any

#: scripted_v1: one hypothesis per cycle, greedy accept-if-score-improves.
SCRIPTED_V1: list[dict[str, Any]] = [
    {"name": "c0-defaults", "patch": {}, "notes": "baseline defaults"},
    {"name": "c1-track-2.5", "patch": {"reward": {"weights": {"tracking_lin_vel": 2.5}}},
     "notes": "stronger velocity tracking"},
    {"name": "c2-smooth", "patch": {"reward": {"weights": {"action_rate": -0.03}}},
     "notes": "penalise jerky targets more"},
    {"name": "c3-stage1", "patch": {"curriculum_stage": 1}, "notes": "add turning commands"},
    {"name": "c4-wide-dr", "patch": {"dr": {"kp": [20.0, 60.0], "latency_steps": [0, 4]}},
     "notes": "wider servo/latency randomisation for robustness"},
    {"name": "c5-longer", "patch": {"ppo": {"num_timesteps": 60_000_000}}, "notes": "1.5x budget on the best"},
]

STRATEGIES = {"scripted_v1": SCRIPTED_V1}
