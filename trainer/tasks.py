"""The task catalogue: what GLM can train, and which gate suite judges each task.

A run's config declares its ``task``; the task declares its ``suite``; nobody types a
suite name by hand. ``GET /tasks`` serves this to the LLM together with the store's
answer to "is the prerequisite satisfied, and which run do I warm-start from".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    name: str
    goal: str
    suite: str
    aliases: tuple[str, ...] = ()
    #: tasks that must have an all-gates-pass run before this one is worth training
    prerequisites: tuple[str, ...] = ()
    #: config paths worth touching for THIS task (shown to the LLM)
    config_keys: tuple[str, ...] = ()
    #: firmware gait replayed on the suite's protocol as the bar, or None
    baseline: str | None = "opencat_trF"
    #: the config patch that puts a run on this task (merged under the LLM's own patch)
    config_patch: dict[str, Any] = field(default_factory=dict)


TASKS: dict[str, Task] = {
    "flat_walk": Task(
        name="flat_walk",
        goal="walk forward on flat ground at a commanded 0.05-0.20 m/s for 10 s without falling",
        suite="flat_v1",
        aliases=("walk", "flat", "forward", "walking", "trot"),
        config_keys=("reward.weights.tracking_lin_vel", "reward.weights.energy", "reward.weights.action_rate",
                     "reward.tracking_sigma", "curriculum_stage", "dr.friction", "dr.kp"),
        config_patch={"task": "flat_walk", "terrain": {"level": 0}},
    ),
    "slope_up": Task(
        name="slope_up",
        goal="walk UP an 8 degree incline at 0.10 m/s for 10 s without falling or sliding sideways",
        suite="slope_v1",
        aliases=("slope", "incline", "ramp", "uphill", "hill", "climb"),
        prerequisites=("flat_walk",),
        config_keys=("terrain.slope_deg", "reward.weights.orientation", "reward.weights.base_height",
                     "reward.weights.slope_progress", "reward.weights.tracking_lin_vel", "ppo.num_timesteps"),
        config_patch={"task": "slope_up", "terrain": {"level": 1}},
    ),
    "rough_walk": Task(
        name="rough_walk",
        goal="walk over a field of 12 mm rocks/edges at 0.12 m/s without stumbling or falling",
        suite="rough_v1",
        aliases=("rough", "rocks", "edges", "rubble", "uneven", "obstacles", "terrain"),
        prerequisites=("slope_up",),
        config_keys=("terrain.box_height_m", "terrain.n_boxes", "reward.weights.foot_clearance",
                     "reward.weights.stumble", "reward.weights.base_height", "reward.weights.stall", "ppo.num_timesteps"),
        config_patch={"task": "rough_walk", "terrain": {"level": 2}},
    ),
    # ── fun moves (config + suite only; the servo-safety fragment guards every one) ──
    "spin": Task(
        name="spin",
        goal="turn in place at a commanded yaw rate (0.5 rad/s) without the torso drifting",
        suite="spin_v1",
        aliases=("pirouette", "turn", "rotate", "twirl", "spin in place"),
        prerequisites=("flat_walk",),
        config_keys=("reward.weights.tracking_ang_vel", "reward.weights.feet_slip", "reward.ang_tracking_sigma",
                     "commands.wz", "reward.weights.stall"),
        baseline="opencat_trL",
        config_patch={"task": "spin", "terrain": {"level": 0},
                      "commands": {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [-0.8, 0.8]},
                      "reward": {"weights": {"tracking_ang_vel": 2.0, "feet_slip": -0.02, "stall": -0.05}}},
    ),
    "statue": Task(
        name="statue",
        goal="stand still and stay upright while being shoved (0.3-1.0 m/s kicks every 1.5-3 s)",
        suite="statue_v1",
        aliases=("push recovery", "balance", "stand still", "shove", "hold", "brace"),
        prerequisites=("flat_walk",),
        config_keys=("reward.weights.stand_still", "reward.weights.orientation", "reward.weights.base_height",
                     "dr.push_vel", "dr.push_interval_s", "reward.weights.stall"),
        baseline="stand",
        config_patch={"task": "statue", "terrain": {"level": 0},
                      "commands": {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [0.0, 0.0]},
                      "dr": {"push_enabled": True, "push_vel": [0.3, 1.0], "push_interval_s": [1.5, 3.0]},
                      "reward": {"weights": {"stand_still": -1.0, "orientation": -4.0, "base_height": -2.0,
                                             "feet_air_time": 0.0, "feet_slip": 0.0, "stall": -0.05}}},
    ),
    "backward_walk": Task(
        name="backward_walk",
        goal="walk backwards at a commanded -0.05 to -0.20 m/s in a straight line",
        suite="backward_v1",
        aliases=("backward", "backwards", "reverse", "retreat", "back up"),
        prerequisites=("flat_walk",),
        config_keys=("reward.weights.tracking_lin_vel", "commands.vx", "reward.weights.stall"),
        baseline="opencat_bkF",
        config_patch={"task": "backward_walk", "terrain": {"level": 0},
                      "commands": {"vx": [-0.20, -0.05], "vy": [0.0, 0.0], "wz": [0.0, 0.0]},
                      "reward": {"weights": {"tracking_lin_vel": 2.0, "stall": -0.05}}},
    ),
}


def task_name_for(config: dict[str, Any]) -> str:
    return str(config.get("task") or "flat_walk")


def default_suite_for(task: str) -> str:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return TASKS[task].suite


def catalogue() -> list[dict[str, Any]]:
    return [{"task": t.name, "goal": t.goal, "suite": t.suite, "aliases": list(t.aliases),
             "prerequisites": list(t.prerequisites), "config_keys": list(t.config_keys),
             "baseline": t.baseline, "config_patch": t.config_patch} for t in TASKS.values()]
