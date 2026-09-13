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
        # the gait has to be RESHAPED (lift the swing foot 12 mm), not tuned: the terms that teach it are on
        # by default (rescaled foot_clearance, stumble, a longer feet_air_time) and the budget is 30M, because
        # 10-15M warm steps left r9/r10/h3 shuffling at ~1 mm swing clearance
        config_patch={"task": "rough_walk", "terrain": {"level": 2},
                      "reward": {"weights": {"foot_clearance": -2.0, "stumble": -0.5, "feet_air_time": 0.3, "stall": -0.05}},
                      "ppo": {"num_timesteps": 30_000_000}},
    ),
    "house_walk": Task(
        name="house_walk",
        goal="ONE policy for the whole house: walk forward, backward and turn on flat, sloped (either direction), "
             "rocky and rocky-sloped ground, on any friction, carrying up to 100 g, while being bumped, and stand "
             "still on command on all of them (house_v1 judges nine scenes at once)",
        suite="house_v1",
        aliases=("house", "everything", "all terrain", "all-terrain", "general", "dynamic", "mixed terrain",
                 "any floor", "carpet", "rug edge", "cable"),
        prerequisites=("slope_up",),
        config_keys=("terrain.slope_share", "terrain.box_share", "terrain.box_height_m", "terrain.slope_deg",
                     "reward.weights.foot_clearance", "reward.weights.stumble", "reward.weights.orientation",
                     "reward.weights.tracking_lin_vel", "reward.weights.tracking_ang_vel", "reward.weights.stand_still",
                     "dr.friction", "dr.payload_g", "dr.push_vel", "commands.vx", "commands.wz", "ppo.num_timesteps"),
        config_patch={"task": "house_walk", "terrain": {"level": 4}, "curriculum_stage": 2,
                      # the full command box: backward, sideways, turning; zero commands come from ZERO_CMD_PROB
                      "commands": {"vx": [-0.15, 0.20], "vy": [-0.08, 0.08], "wz": [-0.6, 0.6]},
                      # the house: hardwood to rug (0.3-1.2), a 100 g payload, a weak battery, late IMU, bumps
                      "dr": {"friction": [0.3, 1.2], "payload_g": [0.0, 100.0], "forcerange": [0.15, 0.35],
                             "latency_steps": [0, 4], "push_enabled": True, "push_vel": [0.1, 0.4],
                             "push_interval_s": [3.0, 6.0]},
                      "reward": {"weights": {"foot_clearance": -0.5, "stumble": -0.5, "energy": -0.05, "stall": -0.05}}},
    ),
    "stairs_walk": Task(
        name="stairs_walk",
        goal="climb a flight of three 18 mm steps (80 mm treads), cross the landing and walk down the other side "
             "at 0.12 m/s without falling (stairs_v1)",
        suite="stairs_v1",
        aliases=("stairs", "stair", "staircase", "steps", "step up", "curb", "kerb", "climb stairs", "doorstep", "ledge"),
        prerequisites=("rough_walk",),
        config_keys=("terrain.stair_rise_m", "terrain.stair_steps", "terrain.stair_tread_m", "terrain.stair_profile",
                     "reward.weights.foot_clearance", "reward.weights.feet_air_time", "reward.weights.stumble",
                     "reward.weights.base_height", "reward.weights.orientation", "reward.weights.lin_vel_z",
                     "reward.weights.stall", "ppo.num_timesteps"),
        # level 5 = 1-4 steps of 6-22 mm, up and down; the rough champion's swing terms carry over (a same-task
        # warm start keeps the parent's values -- these are the cold-start defaults). Terrain-relative base_height
        # reads the MEAN level under the feet, so the torso is asked to climb half a riser at a time.
        config_patch={"task": "stairs_walk", "terrain": {"level": 5},
                      "reward": {"weights": {"foot_clearance": -2.0, "stumble": -0.5, "feet_air_time": 2.0,
                                             "feet_slip": -2.0, "base_height": -2.0, "stall": -0.05}},
                      "ppo": {"num_timesteps": 30_000_000}},
    ),
    "rubble_walk": Task(
        name="rubble_walk",
        goal="walk over a dense field of 20 mm rocks (32 boxes, 7 cm apart, any yaw, no bare floor between them) "
             "at 0.12 m/s without stumbling or falling (rubble_v1)",
        suite="rubble_v1",
        aliases=("rubble", "boulders", "more rocks", "harder rocks", "dense rocks", "big rocks", "gravel", "scree"),
        prerequisites=("rough_walk",),
        config_keys=("terrain.box_height_m", "terrain.n_boxes", "terrain.box_spacing_m", "terrain.box_size_m",
                     "reward.weights.foot_clearance", "reward.weights.feet_air_time", "reward.weights.stumble",
                     "reward.weights.base_height", "reward.weights.stall", "ppo.num_timesteps"),
        # level 6 = every box live, 10-25 mm; the swing terms that lifted the rough gait (r12/r13) are the defaults
        config_patch={"task": "rubble_walk", "terrain": {"level": 6},
                      "reward": {"weights": {"foot_clearance": -2.0, "stumble": -0.5, "feet_air_time": 2.0,
                                             "feet_slip": -2.0, "stall": -0.05}},
                      "ppo": {"num_timesteps": 30_000_000}},
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
    "ball_balance": Task(
        name="ball_balance",
        goal="balance atop an unanchored free-rolling basketball (r=12cm, m=0.6kg) for 10 s without falling",
        suite="ball_v1",
        aliases=("ball", "basketball", "rolling ball", "sphere", "ball balance"),
        prerequisites=("flat_walk",),
        config_keys=("reward.weights.stand_still", "reward.weights.orientation", "reward.weights.base_height",
                     "reward.weights.feet_slip", "reflex.arm", "reflex.max_trim_deg", "ppo.num_timesteps"),
        baseline="stand",
        config_patch={"task": "ball_balance", "terrain": {"level": 0},
                      "commands": {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [0.0, 0.0]},
                      "reward": {"weights": {"stand_still": -1.0, "orientation": -5.0, "base_height": -3.0,
                                             "feet_air_time": 0.0, "feet_slip": -1.0, "stall": -0.05}},
                      "ppo": {"num_timesteps": 3000000}},
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
