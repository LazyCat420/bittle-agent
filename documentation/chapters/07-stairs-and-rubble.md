---
part: RL walking trainer
status: in-progress
updated: 2026-09-10
---

# Stairs and rubble: two harder floors, and the height reference the stairs broke

## The question

The rough_v1 champion (r13, 0.60 m over 12 mm rocks, 13/15 gates) shuffles over edges instead of
stepping. The operator asked for training runs that teach it stairs, whether other obstacles would
teach it more, and for a rockier field. This chapter is what was built (bittle-agent `7f599c2`), what
it proved before any training ran, and the first two runs.

## Stairs are the same 32 boxes

The terrain XML already carries 32 parked boxes placed per env through `body_pos` / `geom_size`
(chapter 04) and re-posed in the Warp data at every reset (chapter 06). A staircase is those boxes
used differently: `terrain.kind: stairs` lays full-width step boxes along +x from `field_start_m`,
tops rising by one riser per tread, then a landing, then the mirror flight down. Slots are fixed so the
vmapped sampler stays shape-stable: 0–5 the flight up, 6 the landing, 7–12 the flight down; the last
step down is the floor, so a flight of *n* has 2*n* live boxes.

| knob | level 5 default (training) | stairs_v1 bar |
|---|---|---|
| `stair_steps` | 1–4 per flight (a single curb is the easiest draw) | 3 |
| `stair_rise_m` | 6–22 mm | 18 mm (the app's mini-stairs preset) |
| `stair_tread_m` | 60–120 mm | 80 mm |
| `stair_profile` | `up_down` | `up_down` |
| landing / width | 0.30 m / 0.60 m | same |

Two geometry decisions carry the physics:

- **Every step is buried 20 mm** below the plane (`STAIR_BURY`), whatever its height, so no tread has a
  gap a foot could slip under. A 54 mm landing is therefore a 74 mm tall box with its centre 17 mm
  above the plane.
- **The parked box grew** from 0.15 × 0.15 × 0.02 to **0.15 × 0.30 × 0.10** half-extents. MuJoCo fixes a
  geom's bounding radius at compile time (chapter 04), so the baked size must be the largest any config
  can request: six 35 mm risers plus the burial, 0.60 m wide. Only the two `*_terrain.xml` variants
  changed; `bittle_cpu.xml` is byte-identical and the flat golden fixture still passes.

The `down` profile spawns the robot **on** a platform (the landing slot moved behind `field_start_m`)
and descends from there. That needed a spawn rule both engines lacked: the spawn z is lifted by the
terrain height under it, and a spawn jitter is refused when it would land on a different level (the
old rule refused any non-zero height, which would have refused every platform spawn).

## The height reference the stairs broke

`base_height`, fall termination (`FALL_Z_MIN` = 20 mm) and `body_clearance` all measured the torso
against the terrain height **under the torso point**. On a staircase that height jumps a whole riser the
instant the torso centre crosses an edge: the reward became a step function, and with a 47 mm standing
height any riser over 25 mm would have read as a fall while the front feet were already on the step.

The reference is now `terrain.support_height`: the **mean terrain height under the four feet**. Front
pair on the step, rear pair below reads half a riser; all four on the landing reads the full height;
flat ground reads 0 everywhere, so the flat trainer is bit-identical (golden fixture). One existing
test changed its premise for this reason: a 60 mm slab under a 105 mm stance holds one foot pair at a
time, so its max reference is half the slab (0.005 m), not the slab (0.010 m).

## What was proved before training

| claim | how | result |
|---|---|---|
| step boxes collide on the CPU | trot replay into 3 × 18 mm, foot–step contact count | contacts > 0; the trot is **stopped** by the first riser (support never rises, 0.4 m) |
| step boxes collide under **Warp** (the engine that trains) | front feet spawned on step 1, warp contact table | > 50 contacts in 50 steps; `support_height` reads 9 mm (half a riser) |
| the `down` spawn stands on the platform under Warp | 50 zero-action steps | torso lifted by 54 mm, > 100 platform contacts, no fall, no sink |
| nothing exceeds the baked box | every live half-extent ≤ `PARKED_HALF` | asserted for every profile |
| level 5 / level 6 train | 300k-step PPO smoke under Warp | reward 17 → 223 (stairs), 16 → 240 (rubble), bar eval running |
| flat is untouched | golden fixture `golden_flat_92f2ccb.json` | unchanged |

Firmware baselines on the new suites (`runs/baselines/*/stairs_v1`, `rubble_v1`): the open-loop trot
walks **backwards** on the stairs protocol (−0.40 m p50) and wkF −0.56 m, so the "beats trot" gates are
degenerate and disabled by the existing rule, exactly as designed.

## Rubble: the rocks made hard

Level 6 and `rubble_v1` answer "add more rocks, make it more complicated" without new geometry: all 32
boxes live, 10–25 mm tall (the bar: 20 mm), 2–8 cm across, 7 cm apart on a 0.5 m strip, any yaw. At
7 cm spacing with 5 cm rocks there is no bare floor to shuffle onto — the swing has to clear an edge
almost as tall as half the standing height. Gates are design targets (0.35 m, 12 mm swing clearance)
until a policy shows what is achievable.

## Which obstacles teach the most (and what is not built)

Ranked by what the 50 Hz blind policy can learn from proprioception alone:

1. **A single curb** (stairs with `stair_steps: [1, 1]`) is the primitive of every step edge; level 5
   draws it a quarter of the time. Train it before a flight.
2. **Stairs up and down in one field** teach two different things: lift the swing (up) and control
   pitch and vertical speed (down). The `descends_the_flight` gate reads the second.
3. **Dense tall rocks** (rubble) force clearance where rough_v1 still allowed shuffling round rocks.
4. Not built, each a small addition now that the spawn lift exists: **gaps** (two raised platforms with a
   trench between), a **balance beam** (one narrow long box), **friction patches** per box (ice / rug),
   **stairs on a slope**, and a `stairs_share` in the level-4 house mixture once s1 passes. Moving
   obstacles would need mocap bodies and are not planned.

## Runs

All warm-started from r13 (`20260910-171039-9a4d1b`), the rough_v1 champion, 30M steps each
(~13 min at 37k steps/s).

### s1: the first stairs run stops at the first riser

| gate | s1 | bar |
|---|---|---|
| fall_rate | **0.00** | ≤ 0.10 |
| climbs_the_flight | 0.009 m | ≥ 0.054 m |
| descends_the_flight | 0.000 m | ≥ 0.054 m |
| forward_distance_p50 | 0.193 m | ≥ 0.80 m |
| progress_ratio | 0.161 | ≥ 0.50 |
| stumble_rate | **0.00** | ≤ 0.30 |
| body_clearance | **37 mm** | ≥ 10 mm |
| servo safety (6 gates) | **all pass** (0.27 W, 3.0 rad/s p99.5, 0.0035 stall) | — |

11/15. Read the per-episode numbers rather than the medians: `stuck_x` is 0.20–0.23 m in 19 of 20
episodes, and the first riser stands at 0.15 m plus up to 40 mm of spawn jitter. The climb column is
0.009 m in 13 episodes and 0.018 m in 7 — that is **one foot pair on step 1 with the rear pair still
on the floor** (the support height is the mean under the four feet), and in the 0.018 cases all four
feet made it up one step. The robot then stands there for 6.5 s of a 10 s episode.

So the policy is not falling off the stairs or scraping its belly; it simply cannot lift its body over
an 18 mm edge. That is the same 0.5 mm swing clearance chapter 06 ends on, met by a taller obstacle.
Two things to change, and s2 changes both: the first ask is too big (18 mm against a 47 mm standing
height), and the task's `config_patch` reset the parent's swing weights from their ±10 bound to 2 / −2
when the task changed (the `_resolve` rule from chapter 06 only protects a **same-task** warm start).

### b1: cancelled — the dense field overflowed the contact budget

The first rubble run logged `broadphase overflow - please increase nconmax to 52` on every step it
reported (4315 lines). The per-world contact budget was a flat 48 for any boxed terrain; 32 rocks at
7 cm spacing, up to 8 cm across, need more. **Contacts that do not fit are dropped**, which is the
phantom-rock failure of chapter 06 arriving by a different route — so the run was cancelled rather
than allowed to finish and report a number.

`gpu_env.contact_budget_per_world` now returns 96 for a dense field (more than 24 boxes, or spacing
under 9 cm) and for stairs (full-width boxes can sit under all four legs at once), 48 for the
rough_v1 class, 16 for flat ground; `njmax` follows. The test drives 128 worlds for 150 random-action
steps on levels 5 and 6 and fails on any overflow line in the captured output. s1's log has none, so
its benchmark above stands.

**The lesson generalises past this repo:** a solver budget sized for one workload silently degrades
physics on a denser one, and it announces itself only in stderr that nothing was reading. Any config
knob that scales with obstacle count needs a test at the densest setting the config allows.

### s2 (curbs) and b1b (rubble, clean physics): the same verdict from both floors

| measure | s1 (18 mm flight) | s2 (trained on 6–14 mm curbs) | b1b (20 mm rubble) |
|---|---|---|---|
| gates | 11/15 | 11/15 | 11/15 |
| highest level reached | 0.009 m | **0.032 m** | — |
| episodes reaching the top | 0/20 | **2/20** | — |
| distance | 0.193 m | **0.275 m** | 0.079 m |
| swing clearance p50 | 0.11 mm | **1.30 mm** | 0.22 mm |
| stalled seconds per episode | 6.5 s | **0.0 s** | 8.8 s |
| fall rate | 0.00 | 0.10 | 0.00 |

b1b reran the rubble with the corrected contact budget and **zero overflow lines over the whole 30M
steps**, so its number is honest: 0.08 m of a 2.2 m field, stalled for 8.8 s of every 10 s episode,
belly 38 mm clear, nothing stumbling — it simply stands there. s2 is the one real gain: training on
curbs teaches it to get one to three of the three steps, and it never stalls. It still never descends,
and its fall rate rose to 0.10, exactly the gate limit.

**Both floors give the same verdict, and it is not about the obstacles.** The rubble reflection states
the mechanism exactly: `foot_clearance` already carries **29.9 %** of the reward with `feet_air_time`
and `feet_slip` pinned at their ±10 bounds, and the swing still does not come.

## The swing terms cannot buy the first swing

Every incentive to lift a foot is **conditional on a lift already happening**:

| term | when it pays | what a dragging gait collects |
|---|---|---|
| `foot_clearance` | multiplied by a swing mask (airborne **and** foot moving) | nothing — the mask is 0 |
| `feet_air_time` | at `first_contact`, proportional to time airborne | nothing — a foot that never left never lands |
| `feet_slip` | penalises sliding **while in contact** | this one does fire, and it favours standing still |

That is a bootstrapping failure, not a tuning gap: the terms meant to teach the swing are all silent
until the swing exists, so no value of them can produce the first one. It also explains why chapter
06's clearance rescale (×1000) and the ±10 bounds bought only 0.5 mm.

**Before adding anything, check the robot can physically do it.** Sweeping one leg's shoulder and knee
over the policy envelope with the body at stand height:

| measurement | result |
|---|---|
| kinematic ceiling of the foot above the plane | **116 mm** |
| reached through the real actuators (kp 10, ±0.25 N·m) in 0.5 s | **74 mm** |

The 12 mm target is not a physical limit — six times the requirement is available in half a second.
The gradient was missing, not the capability.

### `stance_timeout`

A term that pays **before** a swing exists: for each foot, the seconds it has been planted beyond
`STANCE_MAX_S` (0.3 s, one and a half firmware-trot stride periods) while the robot is **commanded to
move**. It grows while the foot drags and stops the instant it lifts. It is exactly 0 for a gait that
already swings, and exactly 0 under a zero command, so the statue task is untouched. Default weight
0.0, so every existing config is unchanged and the flat golden fixture still passes.

**The counted overdue time is capped at 0.5 s per foot**, and that cap is load-bearing.
`weighted_reward` clips the weighted total at 0, so a penalty large enough to sink the sum yields a
flat zero reward with **no gradient anywhere** — strictly worse than no term. Uncapped, a foot planted
9 s of a 10 s episode scores 8.7 by itself and 34.8 over four feet, burying every positive term at any
usable weight. Capped, the whole term is at most 2.0 per step, so −0.1 costs about a fifth of a
typical step's reward.

> **The general trap:** a reward term gated on the behaviour it is meant to teach cannot teach it, at
> any weight. When a term's share is large and the behaviour is still absent, ask whether the term is
> *reachable* from the current policy before touching its weight — and measure the mechanical ceiling
> with an oracle before assuming the policy is at fault.

### What the term actually did: five runs, one variable at a time

The first two attempts were not clean tests and are recorded as such. **s3** raised the terrain rung
*and* switched the term on, so its regression (0.11 m against s2's 0.275 m) cannot be attributed to
either; setting it up as a controlled test was an error. **s3 and b2 were also dosed far too low** —
the training curve shows `stance_timeout` collecting −10.7 per episode against a total reward of 453,
a **2.4 % share**, and the trainer's own rule is that anything under ~2 % is invisible to the
optimiser. b2, a genuinely single-variable run at −0.1, changed nothing measurable:

| rubble arm | distance | swing clearance | stalled / episode | falls | gates |
|---|---|---|---|---|---|
| b1b — no term | 0.079 m | 0.222 mm | 8.8 s | 0.00 | 11/15 |
| b2 — −0.1 (2.7 % share) | 0.088 m | 0.181 mm | 7.7 s | 0.00 | 11/15 |
| **b3 — −0.5** | **0.188 m** | **0.743 mm** | **0.0 s** | 0.25 | 10/15 |

At −0.5 the term reaches a 9 % share and the behaviour changes: **2.4× the distance, 3.3× the swing
clearance, and the stalls disappear entirely** — 8.8 s of standing still per episode becomes 0.0 s.
The energy proxy rises from 0.16 W to 1.18 W, which is the honest signature of a robot that has
started doing work instead of standing on a rock. It now also falls in a quarter of episodes, which
costs it a gate: having stopped freezing, it meets the 20 mm rubble at speed and goes over.

On stairs, with s2's terrain held **exactly** fixed and only the weight changed:

| stairs arm | distance | climb p50 | worst episode's climb | reached the top | falls |
|---|---|---|---|---|---|
| s2 — no term | 0.275 m | 0.032 m | 0.004 m | 2/20 | 0.10 |
| **s4 — −0.5** | 0.292 m | 0.034 m | **0.027 m** | **4/20** | **0.00** |

The medians barely move; the **floor** rises sharply. Every episode now clears at least a step and a
half instead of some barely leaving the ground, the top is reached twice as often, and the falls are
gone. `stance_timeout` buys consistency and safety on stairs, and buys motion itself on rubble.

> **The dose is the experiment.** The same term at −0.1 and at −0.5 is not a weak and a strong version
> of one result — it is a null and a positive. Read a term's reward *share* from the training curve
> before concluding anything from a run that carries it.

## Nothing was paying for the climb

`slope_progress` rewards height gained by projecting velocity onto the uphill direction, which it
derives from the **gravity tilt**. A staircase is level ground at several heights: its gravity points
straight down, `uphill_xy` is exactly 0, and the term is identically 0 on every stairs protocol. The
task asks the robot to climb and **no term in the reward paid for climbing**.

`climb_progress` is the missing twin: the rate at which the support surface under the four feet rises,
positive only (falling off the flight must never pay), only while commanded to move, and expressed per
second so 25 Hz and 50 Hz agree. It is exactly 0 on flat ground and on a slope, where the support
height never changes, and its default weight is 0.

### Running

| run | id | change from its parent |
|---|---|---|
| s5-stairs-climb-progress-30M | queued | warm from s4; `climb_progress` 5.0 and nothing else |

## Open

- **b3 trades stalling for falling** (0.00 → 0.25 fall rate on 20 mm rubble). That is the next rung's
  problem, not a reason to back the weight off: a robot that moves and sometimes falls is a better
  starting point than one that never moves. Raise `orientation` / `base_height` from b3.
- stairs_v1 and rubble_v1 remain uncalibrated 0.1.0; `climbs_the_flight` / `descends_the_flight` are
  course geometry, not measured targets.
- Nothing yet descends a flight. Going down is a different problem from going up (pitch and vertical
  speed rather than clearance) and likely needs its own rung with the `down` profile.
- `climb_progress` at 5.0 is reasoned, not measured: one 18 mm riser crossed in a control step scores
  0.9 m/s, so 5.0 makes a real climb worth a few steps of tracking. Read its share from the curve.
- **A task change re-applies the task's `config_patch` over the parent** (s1 lost r13's ±10 swing
  weights). The same-task guard from chapter 06 does not cover a cross-task warm start.
- **Restarting the trainer service kills any job in flight**; s2's benchmark was orphaned that way and
  its state restored by hand from a complete report. A new reward field needs a restart, so wait for
  `/health` to show no running or queued jobs first.
- The NAS container is not redeployed: another session holds uncommitted GPU-lock changes in the
  primary checkout and the deploy guard refuses an unidentifiable tree.
