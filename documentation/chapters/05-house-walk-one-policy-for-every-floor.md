---
part: RL walking trainer
status: shipped
updated: 2026-09-08
---

# House walk: one policy for every floor

## The question

Chapter 04 trained one policy per terrain: a slope champion, a rocks run, a spin run, a statue run.
Six winners are six networks, and a robot cannot swap networks mid-stride. The question this chapter
answers: **can one policy be dynamic across the floors of a house** — hardwood, rug edges, a ramp in
either direction, cables, a shove from a chair leg, a 100 g payload, a tired battery — while walking
forward, backward and turning, and standing still on command on all of them? And how do the
"different things it can do" chain together at run time?

## How the skills chain (they mostly do not)

Nine of the ten house skills on the wish list are **conditions or commands, not behaviours**. The
policy is one MLP at 50 Hz whose only inputs are the IMU gravity vector, the gyro, the commanded
(vx, vy, wz) and its own recent joint targets. Friction, a rug edge, a push, a slope, a payload, servo
sag and IMU latency all arrive through that gravity/gyro signal, and if they were in the training
distribution the reaction is baked into the weights and fires the instant the body tilts. Nothing
"triggers when it hits a rock": the rock is felt and the reflex is the policy. Turning, backing up,
sidestepping and stopping are just values of the command vector; the layer above (GLM or a navigation
loop) changes the command at 1 Hz and the reflex layer does the rest.

So integration means **widening one policy's training distribution** until it covers the house, and
**judging it on every floor at once** so nothing is forgotten. Self-righting is the one genuine
behaviour switch (different initial state, different termination, different reward); it stays parked
(needs the `tested` joint tier, chapter 04) and would run under a three-state supervisor: walk →
fallen (gravity vector says so for > 0.3 s) → righting policy → upright and still → walk.

## What was built

**The house mixture (`terrain.level` 4).** `TerrainConfig` gained `slope_share` and `box_share`: each
of the 2048 training envs draws slope-or-not and rocks-or-not independently (0.6 / 0.6), so a batch is
16 % flat, 24 % slope only, 24 % rocks only, 36 % both. The slope points any direction (`slope_yaw_deg`
±180°: downhill is where a 47 mm-tall robot tips onto its nose) at 2–10°, rocks are 3–15 mm, and the
rock field starts **1 m behind the spawn** (`field_start_m` may now be negative) so backward and
sideways commands meet rocks too. The share draws come last in the sampler, so a config with both
shares at 1.0 consumes exactly the random stream it always did; levels 0–3 are untouched and the flat
golden fixture still passes.

**The `house_walk` task.** Terrain level 4, curriculum stage 2 with the full command box (vx −0.15…0.20,
vy ±0.08, wz ±0.6 rad/s), pushes of 0.1–0.4 m/s every 3–6 s, friction 0.3–1.2, payload 0–100 g, servo
torque 0.15–0.35 N·m (a 7.0 V battery reads as a weaker servo), IMU latency 0–4 control steps, and the
clearance / stumble / stall terms on. Prerequisite `slope_up`; warm-started from the rocks champion.

**Scenes suites.** A suite may declare `scenes:` — a list of named protocols over shared defaults —
instead of one `protocol:`. `house_v1` has nine:

| scene | ground | command | extra |
|---|---|---|---|
| flat | flat | 0.12 m/s | the flat_v1 protocol |
| slope_up | 8° incline | 0.10 m/s | the slope_v1 protocol |
| slope_down | the same 8°, walked down | 0.10 m/s | gravity pulls forward |
| rough | 24 × 12 mm rocks (seed 11) | 0.12 m/s | the rough_v1 field |
| rough_slope | 16 × 10 mm rocks on 6° | 0.10 m/s | a door sill on a ramp |
| pushed | flat | 0.12 m/s | 0.4 m/s shove every 2.5 s |
| turn | flat | 0.08 m/s + 0.4 rad/s | doorways and corridors |
| backward_rough | 16 × 8 mm rocks **behind** the spawn | −0.10 m/s | backing out from under the sofa over cables |
| statue_rough | 8 × 8 mm rocks around the spawn | 0 | 0.4 m/s shove every 2.5 s: "wait here" |

`trainer/eval/scenes.py` is the one loop the policy benchmark and the firmware-gait baselines both go
through. Every scene publishes `<scene>/<metric>`; the bare top-level metrics pool the forward-walking
scenes (that is what the score and the "beats the trot" ratio read); the servo-safety fragment reads the
**worst episode of any scene**; `fall_rate_max`, `progress_ratio_min` and the summed stand-still falls
span all nine. Each scene has its own `seed_start` (spaced by 1000) so rollouts never collide and the
3D viewer can replay any scene by seed. A classic suite is one unnamed scene and keeps its bare metric
names, so nothing about the earlier suites changed. 30 gates, `0.1.0-uncalibrated`: the flat / slope /
rough scenes carry their own suites' bars, the rest are design targets (nothing in the firmware walks
downhill on command, turns while walking or reverses over rocks).

**Campaign.** `house_v1` in `trainer/campaign/strategies.py`: the rocks champion warm-started onto the
mixture, then one tuning rung. A `best_of:<task>` warm start the campaign has not produced yet is now
looked up on the store's leaderboard (the first attempt cold-started because the driver only knew its
own runs). `render_docs.py` renders one clip per scene and a per-scene figure for scenes suites.

## The firmware gaits on the nine scenes (12 episodes each, tested envelope)

| scene | opencat_trF distance / falls | opencat_wkF distance / falls | note |
|---|---|---|---|
| flat | 0.964 m / 0 | 0.834 m / 0 | |
| slope_up | 0.497 m / 0 | 0.653 m / 0 | |
| slope_down | 1.143 m / 0 | 0.903 m / 0 | open loop: gravity speeds it up |
| rough | 0.406 m / 0 | 0.364 m / **6/12** | |
| rough_slope | 0.111 m / 1/12 | 0.152 m / **6/12** | the rocks stop it |
| pushed | 0.913 m / 0 | 0.805 m / 0 | |
| turn | 0.964 m / 0 (yaw rmse 0.53) | 0.834 m / 0 | it cannot turn |
| backward_rough | **+0.964 m** / 0 | +0.834 m / 0 | it cannot reverse: walks forward regardless |
| statue_rough | 0.329 m / 1/12 | 0.228 m / 3/12 | it cannot stand still |

Pooled forward distance 0.913 m (trot) at 3.83 W, peak joint speed 4.35 rad/s. The stand controller
holds on every scene (drift ≤ 14 mm under the shoves). These are the bars the house policy is measured
against: the trot is a fair opponent on flat, slope and rough, and a non-opponent on turn, backward and
statue, which the ratio gates handle by pooling the forward scenes only.

## A defect the first run found in its own reward breakdown

The first house run trained fine (0 falls across all nine scenes) but its reflection said
`base_height` was **99.2 % of the total reward** (−87 032 per episode against −15 on rough_v1) and
`tracking_lin_vel` 0.46 %. Nothing in the config explains a 5000× jump, so it was the environment.
`terrain_height(x, y)` takes the max over the tops of every box whose xy footprint contains the point;
a *parked* box sits 1 m below the floor with 0.15 m half-extents around the origin. With every box
parked — which is what `box_share` produces for 40 % of envs, and which no earlier level ever did — every
entry is "inside" and the max of the parked tops is **−0.97 m**: within 15 cm of the spawn the robot read
as a metre in the air. The height-relative reward, the swing clearance and the critic's height scan all
saw it. Fix: a parked box never counts as ground, and the height is clamped at 0. Two regression tests
(at the spawn, and a `box_share` 0 env) fail with the fix reverted. The CPU benchmarks were never
affected (a flat protocol loads the box-free XML), so the run's gate numbers were honest; only its
training signal was.

## Runs (2026-09-08, 2048 envs, MuJoCo Warp on the 3090 Ti, 15M-step warm starts of ≈ 9.5 min each)

Every run is judged on `house_v1@0.1.0-uncalibrated`: 9 scenes × 12 episodes plus 9 stand-still
sub-protocols, 30 gates. Score = gates passed + 0.5·min(pooled distance, 1) − pooled fall rate.

| run | start | gates | falls (all 9 scenes) | pooled forward | energy (worst scene) | peak joint speed | stall | result |
|---|---|---|---|---|---|---|---|---|
| h0-house-warm-15M | warm from r9 (rocks) | 24/30 | 0/84 | 0.75 m | 0.97 W | 2.92 rad/s | 0.003 | the run that exposed the parked-box leak (`base_height` 99 % of reward); superseded |
| h0b-house-warm-15M | warm from r9, after the fix | 24/30 | 0/84 | 0.70 m | 0.96 W | 3.32 rad/s | 0.003 | healthy reward shares (tracking 61 %); stands still on every floor; slow on rocks, backward and turning |
| **h1-house-track-15M** | warm from h0b; tracking 4.0 / 2.5, clearance −2.0 | **25/30** | **0/84** | 0.77 m | 1.45 W | 3.21 rad/s | 0.003 | **the champion**: backward over cables −0.55 m (passes), flat 1.10 m, pushed 1.07 m, turn rmse 0.30 |
| h2-house-clearance-15M | warm from h1; clearance −10 (the bound), air time 0.5, yaw tracking 4.0 | 25/30 | 1/84 | 0.76 m | 1.09 W | 3.26 rad/s | 0.005 | turn rate passes (0.24) but it forgets to walk while turning (0.19 m); one fall standing on rocks under a shove; rocks unchanged |
| h3-house-clearance-rescaled-15M | warm from h1; `foot_clearance` term **rescaled ×1000** in `spec.py`, weight −1.0, air time 0.3 | 24/30 | 0/84 | 0.78 m | 1.28 W | 3.58 rad/s | 0.004 | clearance now 4.6 % of the reward; rocks-on-slope 0.11 → 0.17 m, slope down 0.83 m; rocks still 0.35 m, and one episode pins 3 servos at once (a new servo-safety fail) |

**The champion, scene by scene** (h1, 12 episodes each; the firmware trot on the same ground in brackets):

| scene | distance p50 | falls | tracking rmse | note |
|---|---|---|---|---|
| flat | 1.10 m (0.96) | 0 | 0.035 m/s | faster than the trot at a third of its power |
| slope_up 8° | 0.77 m (0.50) | 0 | 0.035 | |
| slope_down 8° | 0.80 m (1.14, open loop runs away) | 0 | 0.041 | walks down at the commanded speed instead of sliding |
| rough 12 mm | 0.34 m (0.41) | 0 | 0.118 | **stuck on edges**, swing clearance 0.6 mm: the open problem |
| rough_slope 10 mm on 6° | 0.11 m (0.11) | 0 | 0.105 | same |
| pushed 0.4 m/s | 1.07 m (0.91) | 0 | 0.041 | keeps walking through the shoves |
| turn 0.08 m/s + 0.4 rad/s | 0.46 m, yaw rmse 0.30 (trot 0.53) | 0 | 0.032 | turns, 0.05 rad/s over the bar |
| backward_rough 8 mm | **−0.55 m** (trot +0.96: cannot reverse) | 0 | 0.084 | backs over the cables |
| statue_rough + shoves | drift 0.024 m (trot 0.33: cannot stand) | 0 | — | stand-still drift ≤ 0.018 m on all nine floors |

| the champion on every floor | | |
|---|---|---|
| ![](media/rl-training/policy_h1-house-track-15M_flat.gif) | ![](media/rl-training/policy_h1-house-track-15M_slope_up.gif) | ![](media/rl-training/policy_h1-house-track-15M_slope_down.gif) |
| ![](media/rl-training/policy_h1-house-track-15M_rough.gif) | ![](media/rl-training/policy_h1-house-track-15M_rough_slope.gif) | ![](media/rl-training/policy_h1-house-track-15M_pushed.gif) |
| ![](media/rl-training/policy_h1-house-track-15M_turn.gif) | ![](media/rl-training/policy_h1-house-track-15M_backward_rough.gif) | ![](media/rl-training/policy_h1-house-track-15M_statue_rough.gif) |

| the firmware trot on the same floors | | |
|---|---|---|
| ![](media/rl-training/baseline_opencat_trot_house_v1_flat.gif) | ![](media/rl-training/baseline_opencat_trot_house_v1_slope_up.gif) | ![](media/rl-training/baseline_opencat_trot_house_v1_slope_down.gif) |
| ![](media/rl-training/baseline_opencat_trot_house_v1_rough.gif) | ![](media/rl-training/baseline_opencat_trot_house_v1_rough_slope.gif) | ![](media/rl-training/baseline_opencat_trot_house_v1_pushed.gif) |
| ![](media/rl-training/baseline_opencat_trot_house_v1_turn.gif) | ![](media/rl-training/baseline_opencat_trot_house_v1_backward_rough.gif) | ![](media/rl-training/baseline_opencat_trot_house_v1_statue_rough.gif) |

![per-scene falls and progress, every house run against the trot](media/rl-training/scenes_house_v1.png)

![house_v1 gates](media/rl-training/gates_house_v1.png)

![training curves](media/rl-training/training_curves.png)

**Reading it.** The policy *is* dynamic across the floors: one network, zero falls in 84 episodes over
nine grounds, standing still on command on all of them, walking down a slope at the commanded speed
where the open-loop trot runs away, backing over cables the firmware cannot reverse over at all, and
carrying on through 0.4 m/s shoves. The five failing gates are two problems, not five:

1. **Rocks.** On the 12 mm field it walks 0.34 m and then stalls at an edge, with 0 stumbles and 0
   falls — the same shuffle chapter 04 saw, and the mixture did not cure it. The reflection is
   right that `foot_clearance` is invisible (< 0.2 % of the reward), and h2 shows why no patch could
   help: at the weight **bound** (−10) its share is still < 0.2 %. The term was a squared height error
   in metres (12 mm → 1.4 × 10⁻⁴), a thousand times smaller than the base-height term, which is scaled
   by 1000 in `spec.py`. That is a scale defect in the environment, not a weight GLM can fix from the
   config, so it was fixed (one line in `reward_terms`, test-pinned: a shuffling swing now costs
   0.144 per foot) and h3 retrained on it. The term became visible (4.6 % share), rocks-on-slope
   improved 0.11 → 0.17 m, but the median swing clearance stayed under 1 mm and the 12 mm field still
   stops it at 0.35 m; the lifting attempts it does make pinned three servos at once in one episode.
   Fifteen million steps on the mixture is not enough to change the gait's shape; the rocks need
   their own rung (level 2 with the rescaled term, then back to the house) or a longer budget. Until
   then the pooled "beats the trot" ratio (0.85 vs 1.1) is dragged down by the rough scenes.
2. **Turning while walking.** h1 turns at rmse 0.30 rad/s against a 0.25 bar; h2 pushed the yaw weight
   to 4.0 and turned at 0.24 but walked only 0.19 m doing it. The two tracking terms trade against
   each other at the same sigma; a walk-and-turn command needs its own share of the training
   commands (today ⅓ of the box has both non-zero) or a separate sigma.

Everything else — friction 0.3–1.2, the 100 g payload, the weak-servo band, 0–4 steps of IMU latency,
pushes from any direction — is inside the training distribution and shows up as "nothing happened" in
the numbers: the nominal-DR benchmark is the easy case for a policy trained that wide. The DR sweep
(`dr_sweep=true`) is the check that it stays that way and was not run in this wave.

## How to reproduce

```bash
# on the GPU box, trainer service running
python -m trainer.eval.baselines --suite house_v1 --names opencat_trF opencat_wkF stand
python -m trainer.campaign --in-process --strategy house_v1 --out runs/campaign_house.json
MUJOCO_GL=osmesa python trainer/scripts/render_docs.py --runs-dir runs --out documentation/media/rl-training
python trainer/scripts/render_docs.py --ledger documentation/chapters/02-training-run-ledger.md
```

Or ask the agent in training mode for "the house policy": `bittle_list_tasks` returns `house_walk`, and
`bittle_train_and_benchmark(task="house_walk")` runs one rung and reports all nine scenes.

## What is still open

- **Rocks.** The rescaled `foot_clearance` term is now visible but 15M steps did not change the shuffle;
  train the rocks rung again under the new scale (r9/r10's clearance weights were on the old, invisible
  scale) and warm the house from that. The `peak_concurrent_stalls` fail on h3 says the lifting motion
  must also stay inside the servo budget.
- `house_v1` is uncalibrated by design on the scenes no firmware gait can do; freeze it once a policy
  passes and the numbers say what is achievable.
- Walk-and-turn: give combined commands their own share of the command box, or their own sigma.
- `dr_sweep` on the house champion (payload, friction, latency presets) has not been run.
- Self-righting (the real behaviour switch) still needs the `tested` joint tier and its own reset
  distribution and termination; the supervisor sketch above is not built.
- The DR sweep and the multi-command sub-protocol run on the primary (flat) scene only.
- Hardware transfer is unchanged (servo speed measurement, degree-convention translation, chapter 01).
