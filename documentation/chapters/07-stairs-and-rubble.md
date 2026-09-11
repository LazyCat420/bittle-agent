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

Both warm-started from r13 (`20260910-171039-9a4d1b`), 30M steps, task defaults; the task change put
the swing terms back from r13's ±10 bound to 2 / −2.

| run | id | task | result |
|---|---|---|---|
| s1-stairs-warm-30M | `20260910-192939-a61192` | stairs_walk, level 5 | training at the time of writing |
| b1-rubble-warm-30M | `20260910-192939-ae201b` | rubble_walk, level 6 | queued |

Results, gate tables and clips follow in this chapter and the ledger (chapter 02) when the benchmarks land.

## Open

- stairs_v1 and rubble_v1 are uncalibrated 0.1.0: freeze thresholds once a learned policy sets a bar.
- A task change re-applies the task `config_patch` over the parent (s1/b1 lost r13's ±10 swing weights);
  decide whether a cross-task warm start should keep the parent's tuned reward weights.
- The trainer service loads tasks at boot; it was restarted on :8009 for this change.
- The NAS container (docs site + agent) is not redeployed yet: another session holds uncommitted
  GPU-lock changes in the primary checkout and the deploy guard refuses an unidentifiable tree.
