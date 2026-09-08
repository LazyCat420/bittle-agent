"""Scripted config-search strategies for the campaign driver (no LLM).

Each step: ``name``, ``task`` (decides the gate suite), ``patch``, ``notes`` and ``base``:
``None`` = from scratch, ``"best"`` = the best run so far on THIS step's suite,
``"best_of:<task>"`` = the best run of another task (the rung-to-rung warm start).
``stop_on_pass`` ends the campaign early when that step passes every gate.
"""

from __future__ import annotations

from typing import Any

#: scripted_v1: one hypothesis per cycle, greedy accept-if-score-improves, flat ground only.
SCRIPTED_V1: list[dict[str, Any]] = [
    {"name": "c0-defaults", "task": "flat_walk", "base": None, "patch": {}, "notes": "baseline defaults"},
    {"name": "c1-track-2.5", "task": "flat_walk", "base": "best", "patch": {"reward": {"weights": {"tracking_lin_vel": 2.5}}},
     "notes": "stronger velocity tracking"},
    {"name": "c2-smooth", "task": "flat_walk", "base": "best", "patch": {"reward": {"weights": {"action_rate": -0.03}}},
     "notes": "penalise jerky targets more"},
    {"name": "c3-stage1", "task": "flat_walk", "base": "best", "patch": {"curriculum_stage": 1}, "notes": "add turning commands"},
    {"name": "c4-wide-dr", "task": "flat_walk", "base": "best", "patch": {"dr": {"kp": [4.0, 20.0], "latency_steps": [0, 4]}},
     "notes": "wider servo/latency randomisation for robustness"},
    {"name": "c5-longer", "task": "flat_walk", "base": "best", "patch": {"ppo": {"num_timesteps": 60_000_000}},
     "notes": "1.5x budget on the best", "stop_on_pass": True},
]

#: terrain_ladder_v1: flat champion -> wide DR -> turning -> slope (warm) -> slope tune -> rocks (warm) -> rocks DR.
#: Reproduces the LLM's campaign ladder without the LLM; per-suite greedy acceptance.
TERRAIN_LADDER_V1: list[dict[str, Any]] = [
    {"name": "t0-flat-baseline", "task": "flat_walk", "base": None, "patch": {}, "notes": "flat champion from scratch"},
    {"name": "t1-flat-dr", "task": "flat_walk", "base": "best", "patch": {"dr": {"kp": [4.0, 20.0], "latency_steps": [0, 4]},
                                                                           "ppo": {"num_timesteps": 10_000_000}},
     "notes": "widen DR before terrain"},
    {"name": "t2-flat-stage1", "task": "flat_walk", "base": "best", "patch": {"curriculum_stage": 1, "ppo": {"num_timesteps": 10_000_000}},
     "notes": "turning commands"},
    {"name": "t3-slope-warm", "task": "slope_up", "base": "best_of:flat_walk", "patch": {"ppo": {"num_timesteps": 10_000_000}},
     "notes": "warm start the flat champion onto the incline"},
    {"name": "t4-slope-tune", "task": "slope_up", "base": "best",
     "patch": {"reward": {"weights": {"base_height": -2.0, "orientation": -4.0}}, "ppo": {"num_timesteps": 10_000_000}},
     "notes": "terrain-relative height + stronger tilt penalty"},
    {"name": "t5-rough-warm", "task": "rough_walk", "base": "best_of:slope_up",
     "patch": {"reward": {"weights": {"foot_clearance": -0.5, "stumble": -0.5}}, "ppo": {"num_timesteps": 10_000_000}},
     "notes": "slope champion onto rocks, with the clearance and stumble terms switched on"},
    {"name": "t6-rough-dr", "task": "rough_walk", "base": "best", "patch": {"dr": {"friction": [0.4, 1.2]}, "ppo": {"num_timesteps": 10_000_000}},
     "notes": "friction spread on rough ground", "stop_on_pass": True},
]

#: house_v1: the rocks champion warm-started onto the level-4 mixture (flat / slope any direction / rocks / both,
#: the full command box, pushes, friction 0.3-1.2, 100 g payload), then one tuning rung on the same suite.
HOUSE_V1: list[dict[str, Any]] = [
    {"name": "h0b-house-warm-15M", "task": "house_walk", "base": "best_of:rough_walk",
     "patch": {"ppo": {"num_timesteps": 15_000_000}},
     "notes": "H0: the rocks champion warm-started onto the house mixture (terrain level 4, stage-2 commands, pushes, wide DR)"},
    {"name": "h1-house-track-15M", "task": "house_walk", "base": "best",
     "patch": {"reward": {"weights": {"tracking_lin_vel": 4.0, "tracking_ang_vel": 2.5, "foot_clearance": -2.0}},
               "ppo": {"num_timesteps": 15_000_000}},
     "notes": "H1: stronger velocity and yaw-rate tracking plus 4x foot clearance on the mixture (h0 was stuck on the "
              "12 mm edges at 0.5 mm swing clearance and turned at rmse 0.37)",
     "stop_on_pass": True},
]

#: house_v1_rung2: the follow-up rung GLM would run after reading h1's diagnosis — foot_clearance was < 0.2 % of the
#: reward at -2.0 (invisible; the policy still shuffles at 0.6 mm swing clearance and sticks on the 12 mm edges),
#: so it goes to the bound with feet_air_time behind it, and yaw tracking gets one more push (turn rmse 0.30 vs 0.25).
HOUSE_V1_RUNG2: list[dict[str, Any]] = [
    {"name": "h2-house-clearance-15M", "task": "house_walk", "base": "best",
     "patch": {"reward": {"weights": {"foot_clearance": -10.0, "feet_air_time": 0.5, "tracking_ang_vel": 4.0}},
               "ppo": {"num_timesteps": 15_000_000}},
     "notes": "H2: foot_clearance x5 to the bound (-10) + feet_air_time 0.5 so the swing actually lifts over 12 mm edges; "
              "tracking_ang_vel 4.0 for the walk-and-turn scene", "stop_on_pass": True},
]

STRATEGIES = {"scripted_v1": SCRIPTED_V1, "terrain_ladder_v1": TERRAIN_LADDER_V1, "house_v1": HOUSE_V1,
              "house_v1_rung2": HOUSE_V1_RUNG2}
