---
part: RL walking trainer
status: shipped
updated: 2026-09-07
---

# Inclines, rocks and fun moves: terrain and task training

## The question

The flat-ground loop works (chapters 01–03). Can GLM now train the robot to walk **up an incline and over
rocks/edges**, and which other **fun moves** are physically safe for the real Bittle X's servos? Simulation
only for now; nothing may close the path to the real robot (IMU-only actor observation, joint targets
clipped to the conservative `agent` envelope, servo model untouched).

## What was built

**Terrain (`trainer/env/terrain.py`).** Two mechanisms, both expressed through fields MuJoCo Warp
batches per world, so every one of the 2048 training envs gets its own ground:

- **Incline = a tilted world.** The floor stays flat and gravity is rotated per env (`opt.gravity`).
  Every height, vertical-velocity and up-vector quantity in the reward is then measured along the
  terrain normal with no code change, and the IMU gravity vector the policy sees is bit-for-bit what a
  real slope produces. A slope run uses the same XML as a flat run — the cleanest possible warm start.
  The one reward term that was wrong on a slope, `orientation`, now reads the torso up-vector in the
  world frame; on flat ground that is algebraically identical to the old term (test-pinned).
- **Rocks/edges = yaw-only half-buried boxes.** 32 boxes are baked into `bittle_{cpu,gpu}_terrain.xml`,
  parked 1 m below the floor, and placed per env (position, yaw, size). Yaw-only keeps the analytic
  `terrain_height(x, y)` exact (verified against `mj_ray` to < 1 µm over 500 points), and half-burying
  means there is never a gap under a box. Heightfields were rejected: not per-world under Warp,
  flagged NaN-prone by Warp's own notes, and boxes are the feature asked for (flat tops, sharp edges).

**Config.** `terrain.level` is a second curriculum axis next to `curriculum_stage` (0 flat, 1 slope 0–8°,
2 rocks 3–12 mm, 3 rocks on a 0–14° slope); the command axis is untouched, so GLM can say "stage 0
commands, terrain level 2". `task` names what a run trains. New bounded reward weights, all **0 by
default**: `foot_clearance`, `stumble`, `slope_progress`, `stall`. A recorded flat-ground action
sequence from the pre-terrain trainer (HEAD 92f2ccb) is replayed by a test and must reproduce every
reward term to 1e-9: terrain support changed nothing on flat ground.

**Critic.** The value network gets a terrain block (3×3 height scan, per-foot clearance, gravity
direction) the actor never sees. That changed the critic's input width, so the three earlier runs
cannot warm-start: `policy.npz` now carries `privileged_version`, and a mismatch is a clean cold start
with the reason recorded on the run, not a shape error inside brax.

**Gate suites per task.** `trainer/eval/suites/<name>.yaml`, each including the shared
`_servo_safety` fragment. A suite's protocol is a fixed bar (a scalar slope, a seeded rock field with
per-episode spawn jitter, a push schedule) that the CPU evaluator applies **over whatever the run
trained on**. Every gate names the reward term that moves it, so the reflection's weight-share advice
is data, not a Python map.

**Task → suite binding for GLM.** A run's config declares its task; the task decides its suite; GLM
never types a suite name. `bittle_list_tasks` answers "what can I train, is the prerequisite met, and
which run do I warm-start from" from the store. `bittle_train_and_benchmark(task=...)` benchmarks on
the task's own suite; benchmarking on another suite needs `force=true` and never enters the
leaderboard, which now ranks per (suite, newest version). The training protocol contains no suite
name (test-enforced). The scripted campaign `terrain_ladder_v1` reproduces the ladder without the LLM.

## The servo-safety fragment (every suite)

| gate | threshold | why |
|---|---|---|
| joint_saturation_pct | ≤ 5 % | existing |
| action_smoothness | ≤ 4 °/step | = 3.5 rad/s, 70 % of the servo speed cap; the firmware trot measures 3.73 |
| energy_proxy | ≤ 2.0 W | the firmware trot burns 3.79 W (chapter 03) |
| **peak_joint_speed** | ≤ 5.0 rad/s (p99.5 over substeps) | the sim servo's own damping-limited speed (0.25 N·m ÷ 0.05); a faster policy is exploiting the integrator. The P1S no-load speed is only known to lie between 86 and 860 °/s (the vendor table says 0.7 s/60°, Petoi's skill guide 0.07 s/60°) — until a servo is timed, nothing above the sim cap is accepted |
| **stall_fraction** | ≤ 0.02 | share of (control step, joint) pairs pinned at ≥ 90 % of the torque cap while barely moving for the whole 20 ms step. A stalled P1S draws 1.5 A |
| **peak_concurrent_stalls** | ≤ 2 | three stalled servos draw 4.5 A, above the battery's 2 A continuous rating; four draw 6 A, above its 5 A peak |

The firmware gaits replayed in the same sim: trot p99.5 joint speed 4.2 rad/s, stall fraction 0.000;
walk 4.5 rad/s, 0.002, at most 1 stalled joint. Substep-level stall counting was tried first and flagged
4–6 "stalled" servos in every firmware gait: transients at gait reversals, not stalls. Hence per control step.

## Calibration: the firmware gaits replayed on each terrain (20 episodes, tested envelope)

| gait | suite | distance p50 | climb | falls | energy | note |
|---|---|---|---|---|---|---|
| opencat_trF | flat_v1 | 0.964 m | — | 0/20 | 3.79 W | |
| opencat_trF | slope_v1 (8°) | 0.497 m | 0.069 m | 0/20 | 3.68 W | slower uphill, never falls |
| opencat_wkF | slope_v1 (8°) | 0.653 m | 0.091 m | 0/20 | 2.97 W | the walk climbs better than the trot |
| opencat_trF | rough_v1 (24 × 12 mm) | 0.381 m | — | 0/20 | 3.68 W | gets stuck on edges, does not fall |
| opencat_wkF | rough_v1 | 0.386 m | — | 12/20 | 2.79 W | |

`slope_v1` is frozen at **1.0.0** from these numbers (distance ≥ 1.1 × trot = 0.55 m, climb ≥ 0.076 m,
falls ≤ 0.10, energy ≤ half the trot's). `rough_v1` stays **0.2.0-uncalibrated**: distance and falls
come from the trot, but foot clearance and stumble rate have no firmware reference — the trot shuffles
(0.7 mm swing peak) and never lifts a foot over anything — so those stay design targets until a learned
policy shows what is achievable. A baseline that cannot do a task at all (travels < 5 cm or falls in
≥ 90 % of episodes) is treated as absent: its ratio gates are not evaluated rather than trivially passed.

## Three physics defects found on the way

1. **The mesh model's chassis collided with the tucked legs.** MuJoCo collides the convex hull of each
   mesh; the chassis hull fills the concave underside where the legs fold. The firmware `wkF` and `crF`
   gaits — which the real robot walks — penetrated the chassis in 64/116 and 103/103 frames (up to
   20 mm). A leg pushed into that phantom wall pins its shoulder servo at the torque cap, i.e. it stalls.
   Fix: contact excludes between the chassis bodies and the knee/shank bodies in the mesh variants
   (the primitive GPU model never had leg–torso collision). Test: zero self-contacts across both gaits.
2. **A geom moved at run time is invisible to collision.** The first rock implementation moved parked
   box geoms via `geom_pos`. The analytic height, `mj_ray` and the privileged observation all saw the
   boxes; the trot walked through them with zero foot–box contacts and a distance identical to flat
   ground. MuJoCo computes each geom's bounding radius and per-body BVH at compile time and the
   broadphase trusts them forever. Boxes are now one static child body each, placed via `body_pos` /
   `body_quat` (re-posed every step), baked at the largest size a config may request. The test that
   catches this is a contact count during a replay, not a ray query.
3. **Stall and clearance definitions.** See the fragment table above; foot clearance is the sphere
   bottom's height above the local terrain, per swing peak — the FK site swings 10 mm relative to the
   rubber paw and read negative during perfectly good swings.

## The fun-move catalogue (sim measurements, `bittle_cpu.xml`)

The robot is torque-rich and speed-poor: standing costs 0.023 N·m (9 % of one servo), the entire
reachable static posture envelope peaks at 17 %. Torque never binds; joint speed and self-collision do.

| move | verdict | the number |
|---|---|---|
| spin in place (`spin_v1`) | shipped | firmware trL turns 0.15 rad/s while drifting 0.44 m; 0.5 rad/s has 3× headroom |
| statue / push recovery (`statue_v1`) | shipped | stand peaks at 9 % torque; pushes are the training mechanism already in DR |
| backward walk (`backward_v1`) | shipped | firmware bkF fits the agent envelope; its replay direction aliases with cadence (forward at 2 rows/step, backward at 1) |
| bow / stretch / paw wave | wave 2 | z 17–55 mm, pitch −13…+18°, all ≤ 17 % torque; needs a per-task fall rule |
| hop / pronk | deferred | 1.9 mm apex at the 5 rad/s cap; 20 mm needs 0.065 s/60°, exactly the disputed servo spec |
| self-righting | parked | needs the `tested` joint tier (leg folds to 17 mm vs 59 mm in `agent`); user chose the agent tier everywhere |
| backflip | rejected | 38 rad/s knee rate (7.6× the cap); landing = 5.4× stall torque through the gears |
| handstand | rejected | pendulum time constant 0.10 s vs up to 60 ms of control latency; 115° joints, 90° pitch |
| push-up, buttUp/rest holds | rejected | 0.9 mm of amplitude; 4 / 2 shoulders stalled at the cap = 6 A / 3 A |

## Runs

All wave-1 runs on 2026-09-07, 2048 envs, MuJoCo Warp on the 3090 Ti; every child is a 10M-step
warm start (≈ 6 min) from the new flat parent unless noted. Gate counts are on each run's own suite.

| run | task / suite | start | gates | distance p50 | falls | energy | peak joint speed | stall | result |
|---|---|---|---|---|---|---|---|---|---|
| r4-flat-parent-critic-v2-40M | flat_walk / flat_v1@1.2.0 | cold, 40M, 18 min | **18/18** | 1.07 m | 0/20 | 0.89 W | 3.13 rad/s | 0.002 | the new parent; passes the five servo-safety gates |
| r5-slope-warm-10M | slope_up / slope_v1@1.0.0 | warm from r4 | **16/16** | 0.87 m up 8° (climb 0.122 m) | 0/20 | 0.65 W | 2.79 rad/s | 0.002 | the firmware trot manages 0.50 m on the same slope |
| r9-rough-warm-10M | rough_walk / rough_v1@0.2.0 | warm from r5, clearance + stumble on | 10/15 | 0.33 m | 0/20 | 1.22 W | 3.43 rad/s | 0.004 | gets stuck on the edges (trot: 0.38 m); needs the GLM loop |
| r6-spin-warm-10M | spin / spin_v1@0.1.0 | warm from r4 | 9/11 | drift 0.081 m | 0/20 | 0.29 W | 1.71 rad/s | 0.005 | barely turns (rate rmse 0.43 vs 0.10 bar); 1 mm over the drift bar |
| r7-statue-warm-10M | statue / statue_v1@1.0.0 | warm from r4 | **11/11** | drift 0.022 m under 0.5 m/s shoves | 0/20 | 0.03 W | 1.18 rad/s | 0.001 | stands through every push |
| r8-backward-warm-10M | backward_walk / backward_v1@1.0.0 | warm from r4 | 12/13 | **−0.94 m** (firmware bkF −0.56 m) | 0/20 | 1.84 W | 2.98 rad/s | 0.002 | one near miss: tracking rmse 0.056 vs 0.05 |

Three of the six new tasks pass outright on the first warm start (slope, statue, and the flat parent
itself), one is a near miss (backward), and two are the ones GLM has to work on: **rough terrain** and
**spin**. Every run stays inside the servo-safety fragment: no run exceeded 3.5 rad/s p99.5 joint speed,
0.005 stall fraction or 2 concurrent stalls. Each run's GIF, rendered in its own suite's terrain, is in
the run ledger (chapter 02).

**Reading the rough result.** The rough policy walks 0.33 m before the boxes stop it, with zero falls
and a stumble rate of 0 — it does not trip, it stalls at edges (velocity-tracking RMSE 0.125 m/s) and
its median swing clearance is 1.2 mm, i.e. it still shuffles like the flat parent. That is exactly the
gap the `foot_clearance` and `stumble` terms exist for; the reflection tells GLM which reward shares are
too small to move. Ten minutes of warm-start training at level 2 is not the end of that curriculum.

**Live dashboard.** The bittle-agent UI gained a **📈 Training** tab: the leaderboard per suite with
live progress bars, per-run reward curves and gate bars, config diffs, the reflection, a one-click
rollout replay in the 3D viewer, side-by-side comparison of any runs (curves, gates, diffs), and the
GLM session panel that shows every training cycle GLM runs — hypothesis, patch, progress, result — live
for every viewer (sessions are recorded server-side and replayable).

## What is still open

- `rough_v1` foot-clearance and stumble thresholds are design targets, not calibrated (see above).
- The stumble sensor on the mesh (CPU) model is the knee-servo housing, because the shank mesh
  includes the rubber paw and touches the ground on every stance; on the GPU model it is the shank
  capsule. The training signal is the GPU one; the CPU gate is weak.
- `dual_sim` is skipped on terrain protocols (the GPU replay has no fixed-terrain override yet).
- Hardware transfer is unchanged and still blocked by the servo speed measurement (film one servo doing
  a 60° step at 240 fps) and the firmware degree-convention translation (chapter 01).
