---
part: RL walking trainer
status: shipped
updated: 2026-09-10
---

# The rocks were phantoms: why "keep training" could not fix rough terrain

## The question

The rough_walk policies (r9, r10, the house champion) attempt the 12 mm rocks, never fall, and get
stuck: 0.31–0.33 m in 10 s against a 0.45 m gate, swing clearance 1.2 mm against 8 mm. The operator
asked whether the agentic harness was missing something or whether the robot just needed more
training. This chapter is the audit, the defect it found, and the two training cycles that followed.

## What the run files said before any code was read

| fact | where |
|---|---|
| r10's training curve walked 0.89 m at step 0 and 0.90 m at 10M; reward 872 → 884 (flat) | `runs/…f10da8/curves.jsonl` |
| The benchmark on the same policy read 0.31 m | its rough_v1 report |
| Training level 2 = 20 boxes, heights uniform 3–12 mm; the benchmark = 24 boxes all 12 mm | `config.py` vs `rough_v1.yaml` |
| The rough_walk task set only `terrain.level: 2`; foot_clearance and stumble weights defaulted to 0 | `tasks.py` |
| The reflection only ever said "multiply the weight" | `gates.py` |
| Dual-sim (GPU vs CPU) was skipped on every terrain protocol | `benchmark_cli.py` |

So the optimiser was solving something easier than the exam and the harness could not see it. The
planned fixes were: match the training terrain to the bar, switch the clearance terms on, make the
clearance term one-sided, add a training-time eval on the suite protocol, teach the reflection to
name a train/benchmark mismatch, add stuck diagnostics, and **run dual-sim on rough once before
training anything**. The last item found the real defect.

## The defect: the GPU never collided with the rocks

Running r10's policy on the pinned rough_v1 field in both engines:

| engine | distance | foot-box contacts | foot bottom vs box top |
|---|---|---|---|
| warp (training engine) | 1.00 m | **0** | **13 mm below** |
| CPU MuJoCo (benchmark engine) | 0.29 m | 104 | never below |

mujoco_warp treats a geom whose body is welded to the world as **static**: its world pose is tiled
into `data.geom_xpos` once at `make_data` from the unbatched `MjModel`, where every terrain box is
parked at z = −1, and the kinematics kernel returns early for it ever after. Placing the boxes per
env through the batched model's `body_pos` (the fix for the compile-time bounding-box trap in
chapter 04) moved only the *analytic* terrain: the reward, the critic's height scan, the foot
clearance and the terrain-relative height all saw rocks; the physics did not. Every GPU rough and
house run to date trained on a flat floor with phantom rocks, which is exactly why the curve read
0.9 m while the benchmark read 0.3 m, and why no reward weight could ever help.

Nobody saw it because the only contact-count test ran on the CPU, dual-sim was defined for flat
protocols only, and the reflection blamed reward shares.

**Fix.** `BittleGpuEnv._place_static_boxes`, called in `reset` after `mjx.forward`, writes the per-env
box poses into `data.geom_xpos` / `geom_xmat`; static geoms keep whatever pose sits in `data`, so it
holds for the episode. `pin_terrain(field)` pins a protocol field on an unbatched env for dual-sim.
Two GPU tests now count foot-box contacts on a 12 mm slab and check the batched data poses per env.
After the fix, dual-sim on r10 reads GPU 0.33 m vs CPU 0.31 m (ratio 1.05, was 3.17).

## What the harness gained

- **Rough dual-sim** on every terrain protocol (the GPU replay pins the protocol's field).
- **Bar eval.** During training, brax's `policy_params_fn` rolls the current params out on the task's
  suite protocol (fixed command, protocol terrain, nominal DR); `curves.jsonl` carries
  `bar_distance_x` / `bar_distance_p50` / `bar_fall_rate` next to `distance_x`, the dashboard plots
  both against the benchmark p50. A train line far above the benchmark is now visible at 4 minutes,
  not after the run.
- **Stuck diagnostics.** Per episode `stuck_seconds`, `stuck_x` (where the first stall began) and
  `stuck_limb_steps` (was a shank or thigh on the ground during the stall); rollouts carry
  `limb_contacts` per frame so the viewer can show which leg caught an edge.
- **A reflection that diagnoses instead of multiplying.** It names a TRAIN/BENCH MISMATCH with the
  terrain gap, a switched-OFF term, a heavy term whose gate did not move against the parent (a budget
  problem), the bar eval against the benchmark (an engine disagreement), where the robot got stuck,
  and — after r11 — a heavy clearance term with a sub-2 mm swing (dragging, not swinging).
- **Diagnose tool and training prompt** read `train_vs_bench` and `terrain_gap` first; terrain keys are
  a first-class lever; 30M steps when the terrain or the gait changes, 10M for a weight tweak.
- **Level 2 covers the bar** (24 boxes, 6–15 mm, 0.10 m spacing); rough_walk defaults switch on
  foot_clearance −2.0, stumble −0.5, feet_air_time 0.3, stall −0.05 at 30M; the clearance term is
  one-sided (overshooting 12 mm is free).

## Two cycles on real rocks

| run | change | gates | distance | swing | what happened |
|---|---|---|---|---|---|
| r10 (before) | phantom rocks | 10/15 | 0.31 m | 1.2 mm | trained on a flat floor |
| r11 | real rocks, task defaults, 30M warm from the slope champion | 10/15 | 0.39 m | 0.66 mm | engines agree (0.97); the policy learned to **drag** — clearance carried 17 % of the reward but a term that only scores swings is nearly free when no swing happens; feet_air_time 0.12 %, feet_slip 0.02 % |
| r12 | feet_air_time 2.0 (20×), feet_slip −2.0 (40×), 30M warm from r11 | **13/15** | **0.54 m** | 0.48 mm | passes distance, progress and beats-the-trot (1.03 → 1.4×); 0 falls, 0 stumbles, 0 stuck episodes; still drags (tracking rmse 0.097, clearance 0.48 mm) |

| r11 on the rough_v1 field | r12 on the rough_v1 field |
|---|---|
| ![](media/rl-training/policy_r11-rough-real-rocks-30M.gif) | ![](media/rl-training/policy_r12-rough-swing-incentive-30M.gif) |

The bar eval tracked the benchmark inside both runs (r12: 0.43 m on the GPU protocol at the end of
training, 0.54 m on the CPU benchmark, ratio 1.16). The training curve is finally about the exam.

## Three cycles by GLM

r11 and r12 were driven by hand to prove the engine fix. The harness changes are for GLM, so the loop
was then handed to it: one training-mode session on the deployed agent, "start from r12, call
`bittle_diagnose_run` first and read `train_vs_bench`, `terrain_gap`, `stuck` and the reflection, then
up to three cycles, each warm from the best rough run, a written hypothesis, at most three keys".

| cycle | GLM's diagnosis → patch | gates | distance | swing | what moved |
|---|---|---|---|---|---|
| r13 | dragging line: swing terms invisible → feet_air_time 10, feet_slip −10 (both to the bound), tracking_lin_vel 2.0, 30M warm from r12 | **13/15** | **0.60 m** | 0.57 mm | +0.06 m, rmse 0.097 → 0.092; still drags with both swing terms at the bound (5.2 % / 2.6 % share) |
| r14 | "terms carry weight → budget and terrain": 40M, boxes 10–18 mm, warm from r13 | 11/15 | 0.43 m | 0.25 mm | regressed: 80 % of episodes stalled, one fall — **and GLM found out why**: passing `task="rough_walk"` re-applied the task's stock defaults over the inherited config and silently reset feet_air_time 10 → 0.3 (share 5.2 % → 0.09 %). It saved a research note; the hypothesis was never fairly tested |
| r15 | r13's config with every weight pinned explicitly, 40M warm from r13 | 11/15 | 0.51 m | 0.44 mm | distance holds but two servo-safety gates fail (stall fraction 0.029, 4 concurrent stalls): the extra budget pushed servos into the torque cap |

GLM read the diagnosis correctly at every step: it went to the swing terms first, checked its own
first instinct with a dry-run and reversed it, moved to terrain and budget once the terms carried
weight, returned to the champion after the regression, and diagnosed the r14 collapse from the config
diff rather than from the physics. Its closing recommendation (a stall-aware r13: `stall −0.5`,
`action_scale_deg 4`, 30M warm from r13) targets r15's new failures without losing the swing terms.

Two harness defects came out of the session, both fixed:

- **Same-task warm starts re-applied the task defaults.** `_resolve` in the trainer service merged the
  task's `config_patch` under the caller's patch whenever a task name was given, so a child on the same
  task inherited the stock values, not the parent's tuning. Now the task defaults land only when the
  task CHANGES (the rung); a same-task child keeps its parent's weights. Test: a rough_walk child of a
  rough_walk parent with feet_air_time 10 keeps 10 and shows no reward diff; a house_walk rung still
  gets level 4 and the house reward switches.
- **The reflection would have looped.** After r13 the two swing weights sit at their bounds and the
  swing still measures half a millimetre; the reflection was about to say "raise feet_air_time and
  feet_slip" again. It now says the weights are at their bounds, no patch can buy the swing, and the
  gate needs a reward design change or a calibrated threshold, which is an operator decision.

r13 stays the rough_v1 champion.

| r13 (GLM cycle 1) | r15 (GLM cycle 3) |
|---|---|
| ![](media/rl-training/policy_r13-rough-swing-bound-30M.gif) | ![](media/rl-training/policy_r15-rough-budget-40M-pinned.gif) |

## What is still open

- **Clearance 8 mm is an uncalibrated design target, and the config cannot reach it.** Five cycles
  moved the gait from 0.31 m to 0.60 m without lifting the feet: the policy crosses 12 mm edges by
  dragging a stiff, fast shuffle, and with feet_air_time and feet_slip at their bounds it still does.
  Either the gate is wrong for this robot (the firmware trot lifts 0.7 mm and the tested policies
  never stumble on this field) or the swing reward needs to be a per-swing PEAK bonus rather than a
  per-step penalty. This is an operator decision; the reflection now says so instead of asking for
  another weight change.
- Velocity tracking rmse 0.092 vs 0.08: the drag costs speed; `tracking_sigma` is the untried lever.
- More budget is not free: r15's extra 10M pushed servos into the torque cap (stall gates). The
  servo-safety fragment caught it, which is what it is for.
- The house champion and every earlier rough run were trained on phantom rocks; the house_v1 rung
  needs re-running from r13 on real rocks.
