---
part: RL walking trainer
status: shipped
updated: 2026-09-10
---

# Training run ledger

Every policy training run recorded by the trainer, grouped by the gate suite it is judged on and
newest last within a suite. Generated from `runs/` by `trainer/scripts/render_docs.py --ledger`;
rerun it after training and commit the result. Score = gates passed + 0.5·min(distance, 1) − fall
rate, and it is only comparable WITHIN a suite. Each suite's protocol line comes from its yaml.

## flat_v1 — 4 runs

Protocol `flat_v1@1.2.0`: 20 seeded episodes, 10 s, command [0.12, 0.0, 0.0], flat ground, CPU MuJoCo on the mesh model; 22 gates incl. the servo-safety fragment.

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20260907-135830-42943f`<br>r1-defaults-40M | flat_walk | 2026-09-07T20:58 | — | 40M | 2048 | 15.0 min | 15/15 | 1.13 m | 0.00 | 15.5 | [gif](media/rl-training/policy_r1-defaults-40M.gif) |
| 2 | `20260907-142153-d17a72`<br>r2-energy-fix-15M | flat_walk | 2026-09-07T21:21 | 20260907-135830-42943f | 15M | 2048 | 7.3 min | 15/15 | 1.19 m | 0.00 | 15.5 | [gif](media/rl-training/policy_r2-energy-fix-15M.gif) |
| 3 | `20260907-165840-e406b4`<br>r3-energy-x100-warm-10M | flat_walk | 2026-09-07T23:58 | 20260907-135830-42943f | 10M | 2048 | 6.4 min | 15/15 | 1.13 m | 0.00 | 15.5 | [gif](media/rl-training/policy_r3-energy-x100-warm-10M.gif) |
| 4 | `20260907-192026-85c9be`<br>r4-flat-parent-critic-v2-40M | flat_walk | 2026-09-08T02:20 | 20260907-165840-e406b4 | 40M | 2048 | 18.4 min | 18/18 | 1.07 m | 0.00 | 18.5 | [gif](media/rl-training/policy_r4-flat-parent-critic-v2-40M.gif) |

## slope_v1 — 1 runs

Protocol `slope_v1@1.0.0`: 20 seeded episodes, 10 s, command [0.1, 0.0, 0.0], a 8.0 deg incline (tilted world), CPU MuJoCo on the mesh model; 16 gates incl. the servo-safety fragment.

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `20260907-193948-30a684`<br>r5-slope-warm-10M | slope_up | 2026-09-08T02:39 | 20260907-192026-85c9be | 10M | 2048 | 6.3 min | 16/16 | 0.87 m | 0.00 | 16.4366 | [gif](media/rl-training/policy_r5-slope-warm-10M.gif) |

## rough_v1 — 7 runs

Protocol `rough_v1@0.2.0-uncalibrated`: 20 seeded episodes, 10 s, command [0.12, 0.0, 0.0], 24 x 12 mm boxes (seed 11, spawn jitter 0.06 m), CPU MuJoCo on the mesh model; 15 gates incl. the servo-safety fragment.

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | `20260907-194643-343fc2`<br>r9-rough-warm-10M | rough_walk | 2026-09-08T02:46 | 20260907-193948-30a684 | 10M | 2048 | 7.3 min | 10/15 | 0.33 m | 0.00 | 10.1671 | [gif](media/rl-training/policy_r9-rough-warm-10M.gif) |
| 10 | `20260907-204343-f10da8`<br>r10-rough-clearance-20x | rough_walk | 2026-09-08T03:43 | 20260907-194643-343fc2 | 10M | 2048 | 7.1 min | 10/15 | 0.31 m | 0.00 | 10.1531 | [gif](media/rl-training/policy_r10-rough-clearance-20x.gif) |
| 16 | `20260910-161601-a45b8e`<br>r11-rough-real-rocks-30M | rough_walk | 2026-09-10T23:16 | 20260907-193948-30a684 | 30M | 2048 | 8.3 min | 10/15 | 0.39 m | 0.00 | 10.1965 | — |
| 17 | `20260910-163154-8e82b7`<br>r12-rough-swing-incentive-30M | rough_walk | 2026-09-10T23:31 | 20260910-161601-a45b8e | 30M | 2048 | 8.5 min | 13/15 | 0.54 m | 0.00 | 13.2694 | — |
| 18 | `20260910-171039-9a4d1b`<br>r13-rough-swing-bound-30M | rough_walk | 2026-09-11T00:10 | 20260910-163154-8e82b7 | 30M | 2048 | 8.0 min | 13/15 | 0.60 m | 0.00 | 13.3011 | — |
| 19 | `20260910-172154-94dafc`<br>r14-rough-budget-terrain-40M | rough_walk | 2026-09-11T00:21 | 20260910-171039-9a4d1b | 40M | 2048 | 10.7 min | 11/15 | 0.43 m | 0.05 | 11.164 | — |
| 20 | `20260910-173522-0d6bc6`<br>r15-rough-budget-40M-pinned | rough_walk | 2026-09-11T00:35 | 20260910-171039-9a4d1b | 40M | 2048 | 9.7 min | 11/15 | 0.51 m | 0.00 | 11.2554 | — |

## statue_v1 — 1 runs

Protocol `statue_v1@1.0.0`: 20 seeded episodes, 10 s, command [0.0, 0.0, 0.0], flat ground, CPU MuJoCo on the mesh model; 11 gates incl. the servo-safety fragment.

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | `20260907-194644-ea153a`<br>r7-statue-warm-10M | statue | 2026-09-08T02:46 | 20260907-192026-85c9be | 10M | 2048 | 6.2 min | 11/11 | 0.01 m | 0.00 | 11.006 | [gif](media/rl-training/policy_r7-statue-warm-10M.gif) |

## spin_v1 — 1 runs

Protocol `spin_v1@0.1.0-uncalibrated`: 20 seeded episodes, 10 s, command [0.0, 0.0, 0.5], flat ground, CPU MuJoCo on the mesh model; 11 gates incl. the servo-safety fragment.

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | `20260907-194644-799782`<br>r6-spin-warm-10M | spin | 2026-09-08T02:46 | 20260907-192026-85c9be | 10M | 2048 | 5.3 min | 9/11 | 0.07 m | 0.00 | 9.0365 | [gif](media/rl-training/policy_r6-spin-warm-10M.gif) |

## backward_v1 — 1 runs

Protocol `backward_v1@1.0.0`: 20 seeded episodes, 10 s, command [-0.12, 0.0, 0.0], flat ground, CPU MuJoCo on the mesh model; 13 gates incl. the servo-safety fragment.

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | `20260907-194644-907826`<br>r8-backward-warm-10M | backward_walk | 2026-09-08T02:46 | 20260907-192026-85c9be | 10M | 2048 | 6.8 min | 12/13 | -0.94 m | 0.00 | 11.5323 | [gif](media/rl-training/policy_r8-backward-warm-10M.gif) |

## house_v1 — 5 runs

Protocol `house_v1@0.1.0-uncalibrated`: 9 scenes (flat, slope_up, slope_down, rough, rough_slope, pushed, turn, backward_rough, statue_rough), 12 seeded episodes each, 10 s, CPU MuJoCo on the mesh model; 30 gates incl. the servo-safety fragment (read on the worst scene).

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | `20260907-223545-768cee`<br>h0-house-warm-15M | house_walk | 2026-09-08T05:35 | 20260907-194643-343fc2 | 15M | 2048 | 10.7 min | 24/30 | 0.75 m | 0.00 | 24.3735 | [gif](media/rl-training/policy_h0-house-warm-15M.gif) |
| 12 | `20260907-225101-719350`<br>h0b-house-warm-15M | house_walk | 2026-09-08T05:51 | 20260907-194643-343fc2 | 15M | 2048 | 9.2 min | 24/30 | 0.70 m | 0.00 | 24.35 | [gif](media/rl-training/policy_h0b-house-warm-15M.gif) |
| 13 | `20260907-230139-5a293e`<br>h1-house-track-15M | house_walk | 2026-09-08T06:01 | 20260907-225101-719350 | 15M | 2048 | 9.3 min | 25/30 | 0.77 m | 0.00 | 25.3832 | [gif](media/rl-training/policy_h1-house-track-15M.gif) |
| 14 | `20260907-231254-05b9c3`<br>h2-house-clearance-15M | house_walk | 2026-09-08T06:12 | 20260907-230139-5a293e | 15M | 2048 | 9.8 min | 25/30 | 0.76 m | 0.00 | 25.3794 | [gif](media/rl-training/policy_h2-house-clearance-15M.gif) |
| 15 | `20260907-232614-ba7f65`<br>h3-house-clearance-rescaled-15M | house_walk | 2026-09-08T06:26 | 20260907-230139-5a293e | 15M | 2048 | 9.5 min | 24/30 | 0.78 m | 0.00 | 24.3882 | [gif](media/rl-training/policy_h3-house-clearance-rescaled-15M.gif) |


### Run 1: r1-defaults-40M (`20260907-135830-42943f`)

- **Task / suite:** flat_walk / flat_v1@1.1.0; from scratch
- **Status:** done
- **Hypothesis / notes:** first full run through the service (servo model kp10/0.25Nm)
- **Config vs defaults:**
- `init_from_parent`: True → None
- `reward.weights.foot_clearance`: 0.0 → None
- `reward.weights.slope_progress`: 0.0 → None
- `reward.weights.stall`: 0.0 → None
- `reward.weights.stumble`: 0.0 → None
- `task`: flat_walk → None
- `terrain`: {'kind': 'flat', 'level': 0, 'slope_deg': [0.0, 0.0], 'slope_yaw_deg': [0.0, 0.0], 'slope_share': 1.0, 'box_share': 1.0, 'n_boxes': 0, 'box_height_m': [0.0, 0.0], 'box_size_m': [0.02, 0.06], 'box_spacing_m': 0.12, 'box_yaw_deg': [-45.0, 45.0], 'field_start_m': 0.15, 'field_width_m': 0.4, 'spawn_jitter_m': 0.0} → None
- **Training:** 40M steps, 2048 envs, 900 s, 44,419 steps/s (warp); eval reward 16 → 762
- **Gates:** 15/15; failing: none
- **Reflection:** BENCHMARK PASSED 15/15 gates (score 15.50). Fell in 0/20 episodes. Median distance 1.13 m in 10.0 s. Velocity tracking RMSE 0.037 m/s. 4 gates not evaluated (stage/group/metric). All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 2: r2-energy-fix-15M (`20260907-142153-d17a72`)

- **Task / suite:** flat_walk / flat_v1@1.1.0; from scratch
- **Status:** done
- **Hypothesis / notes:** Fix energy_proxy gate (1.082W > 1.0W): 10x stronger energy penalty (-0.0005 -> -0.005) per gate hint; raise feet_air_time 0.1 -> 0.25 to encourage longer strides / fewer steps so mean power drops without sacrificing speed (protects tight beats_baseline_trot_distance gate at 1.1m). num_timesteps 40M -> 15M for a short cycle.
- **Config vs parent 20260907-135830-42943f:**
- `ppo.num_timesteps`: 40000000 → 15000000
- `reward.weights.energy`: -0.0005 → -0.005
- `reward.weights.feet_air_time`: 0.1 → 0.25
- **Training:** 15M steps, 2048 envs, 436 s, 34,401 steps/s (warp); eval reward 16 → 688
- **Gates:** 15/15; failing: none
- **Reflection:** BENCHMARK PASSED 15/15 gates (score 15.50). Fell in 0/20 episodes. Median distance 1.19 m in 10.0 s. Velocity tracking RMSE 0.040 m/s. 4 gates not evaluated (stage/group/metric). All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 3: r3-energy-x100-warm-10M (`20260907-165840-e406b4`)

- **Task / suite:** flat_walk / flat_v1@1.1.0; warm start from 20260907-135830-42943f
- **Status:** done
- **Hypothesis / notes:** energy weight x100 (its share was 0.05% of reward); warm start from r1; 10M steps
- **Config vs parent 20260907-135830-42943f:**
- `init_from_parent`: None → True
- `ppo.num_timesteps`: 40000000 → 10000000
- `reward.weights.energy`: -0.0005 → -0.05
- **Training:** 10M steps, 2048 envs, 381 s, 26,217 steps/s (warp); eval reward 741 → 761
- **Gates:** 15/15; failing: none
- **Reflection:** BENCHMARK PASSED 15/15 gates (score 15.50). Fell in 0/20 episodes. Median distance 1.13 m in 10.0 s. Velocity tracking RMSE 0.028 m/s. 4 gates not evaluated (stage/group/metric). All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 4: r4-flat-parent-critic-v2-40M (`20260907-192026-85c9be`)

- **Task / suite:** flat_walk / flat_v1@1.2.0; cold start (critic input layout differs (parent privileged_version 1, this trainer 2); cold start)
- **Status:** done
- **Hypothesis / notes:** M0b: the run-3 config retrained from scratch under the terrain-aware critic layout (privileged_version 2); the parent of every terrain run
- **Config vs parent 20260907-165840-e406b4:**
- `ppo.num_timesteps`: 10000000 → 40000000
- `reward.weights.foot_clearance`: None → 0.0
- `reward.weights.slope_progress`: None → 0.0
- `reward.weights.stall`: None → 0.0
- `reward.weights.stumble`: None → 0.0
- `task`: None → flat_walk
- `terrain`: None → {'box_height_m': [0.0, 0.0], 'box_size_m': [0.02, 0.06], 'box_spacing_m': 0.12, 'box_yaw_deg': [-45.0, 45.0], 'field_start_m': 0.15, 'field_width_m': 0.4, 'kind': 'flat', 'level': 0, 'n_boxes': 0, 'slope_deg': [0.0, 0.0], 'slope_yaw_deg': [0.0, 0.0], 'spawn_jitter_m': 0.0}
- **Training:** 40M steps, 2048 envs, 1104 s, 36,242 steps/s (warp); eval reward 14 → 738
- **Gates:** 18/18; failing: none
- **Reflection:** BENCHMARK PASSED 18/18 gates (score 18.50). Fell in 0/20 episodes. Median distance 1.07 m in 10.0 s. Velocity tracking RMSE 0.039 m/s. 4 gates not evaluated (stage/group/metric). All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 5: r5-slope-warm-10M (`20260907-193948-30a684`)

- **Task / suite:** slope_up / slope_v1@1.0.0; warm start from 20260907-192026-85c9be
- **Status:** done
- **Hypothesis / notes:** M3: flat champion warm-started onto the 0-8 deg tilted world (terrain level 1)
- **Config vs parent 20260907-192026-85c9be:**
- `ppo.num_timesteps`: 40000000 → 10000000
- `task`: flat_walk → slope_up
- `terrain.kind`: flat → slope
- `terrain.level`: 0 → 1
- `terrain.slope_deg`: [0.0, 0.0] → [0.0, 8.0]
- **Training:** 10M steps, 2048 envs, 379 s, 26,376 steps/s (warp); eval reward 742 → 770
- **Gates:** 16/16; failing: none
- **Reflection:** BENCHMARK PASSED 16/16 gates (score 16.44). Fell in 0/20 episodes. Median distance 0.87 m in 10.0 s. Velocity tracking RMSE 0.026 m/s. Parent 20260907-192026-85c9be has no slope_v1 benchmark (it was benchmarked on flat_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-192026-85c9be", suite="slope_v1", force=true) if you want the delta. All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 6: r9-rough-warm-10M (`20260907-194643-343fc2`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260907-193948-30a684
- **Status:** done
- **Hypothesis / notes:** M5: slope champion warm-started onto 12 mm rocks with foot_clearance and stumble switched on
- **Config vs parent 20260907-193948-30a684:**
- `reward.weights.foot_clearance`: 0.0 → -0.5
- `reward.weights.stumble`: 0.0 → -0.5
- `task`: slope_up → rough_walk
- `terrain.box_height_m`: [0.0, 0.0] → [0.003, 0.012]
- `terrain.kind`: slope → rough
- `terrain.level`: 1 → 2
- `terrain.n_boxes`: 0 → 20
- `terrain.slope_deg`: [0.0, 8.0] → [0.0, 0.0]
- `terrain.spawn_jitter_m`: 0.0 → 0.05
- **Training:** 10M steps, 2048 envs, 440 s, 22,702 steps/s (warp); eval reward 752 → 749
- **Gates:** 10/15; failing: forward_distance_p50 (0.334 >= 0.45), progress_ratio (0.278 >= 0.4), vel_tracking_rmse (0.125 <= 0.08), foot_clearance (1.225 >= 8.0), beats_baseline_trot_distance (0.878 >= 1.1)
- **Reflection:** BENCHMARK FAILED 10/15 gates (score 10.17). Fell in 0/20 episodes. Median distance 0.33 m in 10.0 s. Velocity tracking RMSE 0.125 m/s. FAIL forward_distance_p50: 0.334 >= 0.45 m (firmware trot 0.381) -> 1.1 x the firmware trot's 0.381 m across this field rounds to 0.42, kept at 0.45; ideal is 1.2 m at 0.12 m/s for 10 s FAIL progress_ratio: 0.278 >= 0.4 ratio (firmware trot 0.317) -> distance / (commanded speed x seconds); the robot is getting stuck on edges FAIL vel_tracking_rmse: 0.125 <= 0.08 m/s (firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel FAIL foot_clearance: 1.225 >= 8.0 mm (firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm). The 'foot_clearance' term is only 0.01% of the total reward, so the optimiser barely sees it: multiply its weight by ~333x (to ~2% share), not by 2-10x. (+1 more failing gates in the table) Parent 20260907-193948-30a684 has no rough_v1 benchmark (it was benchmarked on slope_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-193948-30a684", suite="rough_v1", force=true) if you want the delta.

### Run 7: r7-statue-warm-10M (`20260907-194644-ea153a`)

- **Task / suite:** statue / statue_v1@1.0.0; warm start from 20260907-192026-85c9be
- **Status:** done
- **Hypothesis / notes:** M7: push-resistant stand from the flat champion (pushes 0.3-1.0 m/s)
- **Config vs parent 20260907-192026-85c9be:**
- `commands.vx`: [0.05, 0.2] → [0.0, 0.0]
- `dr.push_enabled`: False → True
- `dr.push_interval_s`: [3.0, 6.0] → [1.5, 3.0]
- `dr.push_vel`: [0.1, 0.3] → [0.3, 1.0]
- `ppo.num_timesteps`: 40000000 → 10000000
- `reward.weights.base_height`: -1.0 → -2.0
- `reward.weights.feet_air_time`: 0.1 → 0.0
- `reward.weights.feet_slip`: -0.05 → 0.0
- `reward.weights.orientation`: -2.0 → -4.0
- `reward.weights.stall`: 0.0 → -0.05
- `reward.weights.stand_still`: -0.2 → -1.0
- `task`: flat_walk → statue
- **Training:** 10M steps, 2048 envs, 370 s, 27,025 steps/s (warp); eval reward 382 → 452
- **Gates:** 11/11; failing: none
- **Reflection:** BENCHMARK PASSED 11/11 gates (score 11.01). Fell in 0/20 episodes. Median distance 0.01 m in 10.0 s. Velocity tracking RMSE 0.025 m/s. Parent 20260907-192026-85c9be has no statue_v1 benchmark (it was benchmarked on flat_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-192026-85c9be", suite="statue_v1", force=true) if you want the delta. The firmware stand cannot do this task (travels 0.01 m, falls 0/20), so there is no baseline to beat on statue_v1; the ratio gates are not evaluated and the absolute distance gate carries the bar. All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 8: r6-spin-warm-10M (`20260907-194644-799782`)

- **Task / suite:** spin / spin_v1@0.1.0-uncalibrated; warm start from 20260907-192026-85c9be
- **Status:** done
- **Hypothesis / notes:** M7: pirouette from the flat champion (wz +-0.8, feet_slip relaxed)
- **Config vs parent 20260907-192026-85c9be:**
- `commands.vx`: [0.05, 0.2] → [0.0, 0.0]
- `commands.wz`: [0.0, 0.0] → [-0.8, 0.8]
- `ppo.num_timesteps`: 40000000 → 10000000
- `reward.weights.feet_slip`: -0.05 → -0.02
- `reward.weights.stall`: 0.0 → -0.05
- `reward.weights.tracking_ang_vel`: 0.5 → 2.0
- `task`: flat_walk → spin
- **Training:** 10M steps, 2048 envs, 316 s, 31,691 steps/s (warp); eval reward 104 → 1232
- **Gates:** 9/11; failing: spin_rate_rmse (0.434 <= 0.1), centre_drift (0.081 <= 0.08)
- **Reflection:** BENCHMARK FAILED 9/11 gates (score 9.04). Fell in 0/20 episodes. Median distance 0.07 m in 10.0 s. Velocity tracking RMSE 0.021 m/s. FAIL spin_rate_rmse: 0.434 <= 0.1 rad/s (firmware trot 0.639) -> raise tracking_ang_vel or loosen ang_tracking_sigma; the firmware turn manages ~0.15 rad/s FAIL centre_drift: 0.081 <= 0.08 m (firmware trot 0.438) -> the torso must stay put: with vx=vy=0 commanded, tracking_lin_vel penalises translation Parent 20260907-192026-85c9be has no spin_v1 benchmark (it was benchmarked on flat_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-192026-85c9be", suite="spin_v1", force=true) if you want the delta.

### Run 9: r8-backward-warm-10M (`20260907-194644-907826`)

- **Task / suite:** backward_walk / backward_v1@1.0.0; warm start from 20260907-192026-85c9be
- **Status:** done
- **Hypothesis / notes:** M7: backward walk from the flat champion (vx -0.20..-0.05)
- **Config vs parent 20260907-192026-85c9be:**
- `commands.vx`: [0.05, 0.2] → [-0.2, -0.05]
- `ppo.num_timesteps`: 40000000 → 10000000
- `reward.weights.stall`: 0.0 → -0.05
- `reward.weights.tracking_lin_vel`: 1.5 → 2.0
- `task`: flat_walk → backward_walk
- **Training:** 10M steps, 2048 envs, 408 s, 24,521 steps/s (warp); eval reward 273 → 776
- **Gates:** 12/13; failing: vel_tracking_rmse (0.056 <= 0.05)
- **Reflection:** BENCHMARK FAILED 12/13 gates (score 11.53). Fell in 0/20 episodes. Median distance -0.94 m in 10.0 s. Velocity tracking RMSE 0.056 m/s. FAIL vel_tracking_rmse: 0.056 <= 0.05 m/s (firmware trot 0.088) -> tighten tracking_sigma or raise tracking_lin_vel Parent 20260907-192026-85c9be has no backward_v1 benchmark (it was benchmarked on flat_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-192026-85c9be", suite="backward_v1", force=true) if you want the delta. The firmware opencat_bkF cannot do this task (travels -0.56 m, falls 0/20), so there is no baseline to beat on backward_v1; the ratio gates are not evaluated and the absolute distance gate carries the bar.

### Run 10: r10-rough-clearance-20x (`20260907-204343-f10da8`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260907-194643-343fc2
- **Status:** done
- **Hypothesis / notes:** Hypothesis: last rough run never falls but shuffles at 1.2mm swing clearance and sticks on edges. foot_clearance share was 0.006% (invisible) -> 20x to -10.0 (max bound). tracking_sigma 0.01 saturated the shaped tracking reward at RMSE 0.125 -> loosen to 0.05 to restore gradient. feet_air_time 0.1 -> 1.0 (10x) to promote swing cycles over edges. Expect foot_clearance, progress_ratio, forward_distance, vel_tracking_rmse to move.
- **Config vs parent 20260907-194643-343fc2:**
- `reward.tracking_sigma`: 0.01 → 0.05
- `reward.weights.feet_air_time`: 0.1 → 1.0
- `reward.weights.foot_clearance`: -0.5 → -10.0
- **Training:** 10M steps, 2048 envs, 426 s, 23,491 steps/s (warp); eval reward 872 → 884
- **Gates:** 10/15; failing: forward_distance_p50 (0.306 >= 0.45), progress_ratio (0.255 >= 0.4), vel_tracking_rmse (0.121 <= 0.08), foot_clearance (1.194 >= 8.0), beats_baseline_trot_distance (0.805 >= 1.1)
- **Reflection:** BENCHMARK FAILED 10/15 gates (score 10.15). Fell in 0/20 episodes. Median distance 0.31 m in 10.0 s. Velocity tracking RMSE 0.121 m/s. FAIL forward_distance_p50: 0.306 >= 0.45 m (parent 0.334, firmware trot 0.381) -> 1.1 x the firmware trot's 0.381 m across this field rounds to 0.42, kept at 0.45; ideal is 1.2 m at 0.12 m/s for 10 s FAIL progress_ratio: 0.255 >= 0.4 ratio (parent 0.278, firmware trot 0.317) -> distance / (commanded speed x seconds); the robot is getting stuck on edges FAIL vel_tracking_rmse: 0.121 <= 0.08 m/s (parent 0.125, firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel. The 'tracking_lin_vel' term already carries 69.1% of the reward and the gate did not move vs the parent: the weight is not the lever -- this needs a longer budget (ppo.num_timesteps 30M: a gait has to be reshaped, not tuned) or a harder training terrain, not another weight change. FAIL foot_clearance: 1.194 >= 8.0 mm (parent 1.225, firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm). The 'foot_clearance' term is only 0.12% of the total reward, so the optimiser barely sees it: multiply its weight by ~17x (to ~2% share), not by 2-10x. (+1 more failing gates in the table) TRAIN/BENCH MISMATCH: the training curve walked 0.90 m per episode (random commands on the run's own terrain) but the suite reads 0.31 m; the run trained on 20 boxes of 3-12 mm (0% at or above the protocol's 12 mm) vs the protocol's 24 boxes all at 12 mm. The optimiser solved an easier field than the exam: raise terrain.box_height_m / terrain.n_boxes (or terrain.slope_deg) to cover the protocol BEFORE touching reward weights.

### Run 11: h0-house-warm-15M (`20260907-223545-768cee`)

- **Task / suite:** house_walk / house_v1@0.1.0-uncalibrated; warm start from 20260907-194643-343fc2
- **Status:** done
- **Hypothesis / notes:** H0: the rocks champion warm-started onto the house mixture (terrain level 4, stage-2 commands, pushes, wide DR) | DEFECT (found from its own reward breakdown): terrain_height leaked the parked-box sentinel (-0.97 m) in every env that drew no boxes, so base_height was 99% of the reward; superseded by h0b after the fix
- **Config vs parent 20260907-194643-343fc2:**
- `commands.vx`: [0.05, 0.2] → [-0.15, 0.2]
- `commands.vy`: [0.0, 0.0] → [-0.08, 0.08]
- `commands.wz`: [0.0, 0.0] → [-0.6, 0.6]
- `curriculum_stage`: 0 → 2
- `dr.friction`: [0.5, 1.1] → [0.3, 1.2]
- `dr.latency_steps`: [0, 3] → [0, 4]
- `dr.payload_g`: [0.0, 30.0] → [0.0, 100.0]
- `dr.push_enabled`: False → True
- `dr.push_vel`: [0.1, 0.3] → [0.1, 0.4]
- `ppo.num_timesteps`: 10000000 → 15000000
- `reward.weights.stall`: 0.0 → -0.05
- `task`: rough_walk → house_walk
- `terrain.box_height_m`: [0.003, 0.012] → [0.003, 0.015]
- `terrain.box_share`: None → 0.6
- `terrain.field_start_m`: 0.15 → -1.0
- `terrain.kind`: rough → rough_slope
- `terrain.level`: 2 → 4
- `terrain.n_boxes`: 20 → 28
- `terrain.slope_deg`: [0.0, 0.0] → [2.0, 10.0]
- `terrain.slope_share`: None → 0.6
- `terrain.slope_yaw_deg`: [0.0, 0.0] → [-180.0, 180.0]
- `terrain.spawn_jitter_m`: 0.05 → 0.08
- **Training:** 15M steps, 2048 envs, 642 s, 23,380 steps/s (warp); eval reward 41 → 366
- **Gates:** 24/30; failing: progress_worst_scene (0.120 >= 0.35), beats_baseline_trot_distance (0.818 >= 1.1), rough_progress (0.320 >= 0.4), rough_slope_progress (0.120 >= 0.35), turn_rate_rmse (0.368 <= 0.25), backward_rough_distance (-0.272 <= -0.35)
- **Reflection:** BENCHMARK FAILED 24/30 gates (score 24.37). Fell in 0/84 episodes. Median distance 0.75 m in 10.0 s. Velocity tracking RMSE 0.063 m/s. FAIL progress_worst_scene: 0.120 >= 0.35 ratio (firmware trot -0.964) -> distance along the commanded direction / (|vx| x seconds), worst moving scene; the robot is stuck somewhere. The 'tracking_lin_vel' term is only 0.46% of the total reward, so the optimiser barely sees it: multiply its weight by ~4x (to ~2% share), not by 2-10x. FAIL beats_baseline_trot_distance: 0.818 >= 1.1 ratio -> the firmware trot replayed on the same nine scenes, pooled forward distance. The 'tracking_lin_vel' term is only 0.46% of the total reward, so the optimiser barely sees it: multiply its weight by ~4x (to ~2% share), not by 2-10x. FAIL rough_progress: 0.320 >= 0.4 ratio (firmware trot 0.338) -> the rough_v1 bar; stuck on the 12 mm edges: raise foot_clearance / stumble 10-100x. The 'tracking_lin_vel' term is only 0.46% of the total reward, so the optimiser barely sees it: multiply its weight by ~4x (to ~2% share), not by 2-10x. FAIL rough_slope_progress: 0.120 >= 0.35 ratio (firmware trot 0.111) -> rocks on a 6 deg ramp; raise foot_clearance and slope_progress. The 'tracking_lin_vel' term is only 0.46% of the total reward, so the optimiser barely sees it: multiply its weight by ~4x (to ~2% share), not by 2-10x. (+2 more failing gates in the table) Parent 20260907-194643-343fc2 has no house_v1 benchmark (it was benchmarked on rough_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-194643-343fc2", suite="house_v1", force=true) if you want the delta.

### Run 12: h0b-house-warm-15M (`20260907-225101-719350`)

- **Task / suite:** house_walk / house_v1@0.1.0-uncalibrated; warm start from 20260907-194643-343fc2
- **Status:** done
- **Hypothesis / notes:** H0: the rocks champion warm-started onto the house mixture (terrain level 4, stage-2 commands, pushes, wide DR)
- **Config vs parent 20260907-194643-343fc2:**
- `commands.vx`: [0.05, 0.2] → [-0.15, 0.2]
- `commands.vy`: [0.0, 0.0] → [-0.08, 0.08]
- `commands.wz`: [0.0, 0.0] → [-0.6, 0.6]
- `curriculum_stage`: 0 → 2
- `dr.friction`: [0.5, 1.1] → [0.3, 1.2]
- `dr.latency_steps`: [0, 3] → [0, 4]
- `dr.payload_g`: [0.0, 30.0] → [0.0, 100.0]
- `dr.push_enabled`: False → True
- `dr.push_vel`: [0.1, 0.3] → [0.1, 0.4]
- `ppo.num_timesteps`: 10000000 → 15000000
- `reward.weights.stall`: 0.0 → -0.05
- `task`: rough_walk → house_walk
- `terrain.box_height_m`: [0.003, 0.012] → [0.003, 0.015]
- `terrain.box_share`: None → 0.6
- `terrain.field_start_m`: 0.15 → -1.0
- `terrain.kind`: rough → rough_slope
- `terrain.level`: 2 → 4
- `terrain.n_boxes`: 20 → 28
- `terrain.slope_deg`: [0.0, 0.0] → [2.0, 10.0]
- `terrain.slope_share`: None → 0.6
- `terrain.slope_yaw_deg`: [0.0, 0.0] → [-180.0, 180.0]
- `terrain.spawn_jitter_m`: 0.05 → 0.08
- **Training:** 15M steps, 2048 envs, 555 s, 27,028 steps/s (warp); eval reward 47 → 490
- **Gates:** 24/30; failing: progress_worst_scene (0.089 >= 0.35), beats_baseline_trot_distance (0.767 >= 1.1), rough_progress (0.274 >= 0.4), rough_slope_progress (0.089 >= 0.35), turn_rate_rmse (0.369 <= 0.25), backward_rough_distance (-0.236 <= -0.35)
- **Reflection:** BENCHMARK FAILED 24/30 gates (score 24.35). Fell in 0/84 episodes. Median distance 0.70 m in 10.0 s. Velocity tracking RMSE 0.062 m/s. FAIL progress_worst_scene: 0.089 >= 0.35 ratio (firmware trot -0.964) -> distance along the commanded direction / (|vx| x seconds), worst moving scene; the robot is stuck somewhere FAIL beats_baseline_trot_distance: 0.767 >= 1.1 ratio -> the firmware trot replayed on the same nine scenes, pooled forward distance FAIL rough_progress: 0.274 >= 0.4 ratio (firmware trot 0.338) -> the rough_v1 bar; stuck on the 12 mm edges: raise foot_clearance / stumble 10-100x FAIL rough_slope_progress: 0.089 >= 0.35 ratio (firmware trot 0.111) -> rocks on a 6 deg ramp; raise foot_clearance and slope_progress (+2 more failing gates in the table) Parent 20260907-194643-343fc2 has no house_v1 benchmark (it was benchmarked on rough_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-194643-343fc2", suite="house_v1", force=true) if you want the delta.

### Run 13: h1-house-track-15M (`20260907-230139-5a293e`)

- **Task / suite:** house_walk / house_v1@0.1.0-uncalibrated; warm start from 20260907-225101-719350
- **Status:** done
- **Hypothesis / notes:** H1: stronger velocity and yaw-rate tracking plus 4x foot clearance on the mixture (h0 was stuck on the 12 mm edges at 0.5 mm swing clearance and turned at rmse 0.37)
- **Config vs parent 20260907-225101-719350:**
- `reward.weights.foot_clearance`: -0.5 → -2.0
- `reward.weights.tracking_ang_vel`: 0.5 → 2.5
- `reward.weights.tracking_lin_vel`: 1.5 → 4.0
- **Training:** 15M steps, 2048 envs, 557 s, 26,951 steps/s (warp); eval reward 1914 → 1983
- **Gates:** 25/30; failing: progress_worst_scene (0.105 >= 0.35), beats_baseline_trot_distance (0.840 >= 1.1), rough_progress (0.286 >= 0.4), rough_slope_progress (0.105 >= 0.35), turn_rate_rmse (0.297 <= 0.25)
- **Reflection:** BENCHMARK FAILED 25/30 gates (score 25.38). Fell in 0/84 episodes. Median distance 0.77 m in 10.0 s. Velocity tracking RMSE 0.058 m/s. FAIL progress_worst_scene: 0.105 >= 0.35 ratio (parent 0.089, firmware trot -0.964) -> distance along the commanded direction / (|vx| x seconds), worst moving scene; the robot is stuck somewhere FAIL beats_baseline_trot_distance: 0.840 >= 1.1 ratio (parent 0.767) -> the firmware trot replayed on the same nine scenes, pooled forward distance FAIL rough_progress: 0.286 >= 0.4 ratio (parent 0.274, firmware trot 0.338) -> the rough_v1 bar; stuck on the 12 mm edges: raise foot_clearance / stumble 10-100x FAIL rough_slope_progress: 0.105 >= 0.35 ratio (parent 0.089, firmware trot 0.111) -> rocks on a 6 deg ramp; raise foot_clearance and slope_progress (+1 more failing gates in the table)

### Run 14: h2-house-clearance-15M (`20260907-231254-05b9c3`)

- **Task / suite:** house_walk / house_v1@0.1.0-uncalibrated; warm start from 20260907-230139-5a293e
- **Status:** done
- **Hypothesis / notes:** H2: foot_clearance x5 to the bound (-10) + feet_air_time 0.5 so the swing actually lifts over 12 mm edges; tracking_ang_vel 4.0 for the walk-and-turn scene
- **Config vs parent 20260907-230139-5a293e:**
- `reward.weights.feet_air_time`: 0.1 → 0.5
- `reward.weights.foot_clearance`: -2.0 → -10.0
- `reward.weights.tracking_ang_vel`: 2.5 → 4.0
- **Training:** 15M steps, 2048 envs, 585 s, 25,636 steps/s (warp); eval reward 2581 → 2567
- **Gates:** 25/30; failing: progress_worst_scene (0.117 >= 0.35), beats_baseline_trot_distance (0.831 >= 1.1), rough_progress (0.286 >= 0.4), rough_slope_progress (0.117 >= 0.35), statue_rough_falls (0.083 <= 0.0)
- **Reflection:** BENCHMARK FAILED 25/30 gates (score 25.38). Fell in 0/84 episodes. Median distance 0.76 m in 10.0 s. Velocity tracking RMSE 0.057 m/s. FAIL progress_worst_scene: 0.117 >= 0.35 ratio (parent 0.105, firmware trot -0.964) -> distance along the commanded direction / (|vx| x seconds), worst moving scene; the robot is stuck somewhere FAIL beats_baseline_trot_distance: 0.831 >= 1.1 ratio (parent 0.840) -> the firmware trot replayed on the same nine scenes, pooled forward distance FAIL rough_progress: 0.286 >= 0.4 ratio (parent 0.286, firmware trot 0.338) -> the rough_v1 bar; stuck on the 12 mm edges: raise foot_clearance / stumble 10-100x FAIL rough_slope_progress: 0.117 >= 0.35 ratio (parent 0.105, firmware trot 0.111) -> rocks on a 6 deg ramp; raise foot_clearance and slope_progress (+1 more failing gates in the table)

### Run 15: h3-house-clearance-rescaled-15M (`20260907-232614-ba7f65`)

- **Task / suite:** house_walk / house_v1@0.1.0-uncalibrated; warm start from 20260907-230139-5a293e
- **Status:** done
- **Hypothesis / notes:** H3: foot_clearance term rescaled x1000 in spec.py (it was 1.4e-4 per 12 mm miss and < 0.2 % of the reward at the weight bound); weight -1.0 under the new scale + feet_air_time 0.3, warm from h1
- **Config vs parent 20260907-230139-5a293e:**
- `reward.weights.feet_air_time`: 0.1 → 0.3
- `reward.weights.foot_clearance`: -2.0 → -1.0
- **Training:** 15M steps, 2048 envs, 570 s, 26,303 steps/s (warp); eval reward 1922 → 1905
- **Gates:** 24/30; failing: progress_worst_scene (0.173 >= 0.35), beats_baseline_trot_distance (0.851 >= 1.1), rough_progress (0.288 >= 0.4), rough_slope_progress (0.173 >= 0.35), turn_rate_rmse (0.288 <= 0.25), peak_concurrent_stalls (3.000 <= 2)
- **Reflection:** BENCHMARK FAILED 24/30 gates (score 24.39). Fell in 0/84 episodes. Median distance 0.78 m in 10.0 s. Velocity tracking RMSE 0.058 m/s. FAIL progress_worst_scene: 0.173 >= 0.35 ratio (parent 0.105, firmware trot -0.964) -> distance along the commanded direction / (|vx| x seconds), worst moving scene; the robot is stuck somewhere FAIL beats_baseline_trot_distance: 0.851 >= 1.1 ratio (parent 0.840) -> the firmware trot replayed on the same nine scenes, pooled forward distance FAIL rough_progress: 0.288 >= 0.4 ratio (parent 0.286, firmware trot 0.338) -> the rough_v1 bar; stuck on the 12 mm edges: raise foot_clearance / stumble 10-100x FAIL rough_slope_progress: 0.173 >= 0.35 ratio (parent 0.105, firmware trot 0.111) -> rocks on a 6 deg ramp; raise foot_clearance and slope_progress (+2 more failing gates in the table)

### Run 16: r11-rough-real-rocks-30M (`20260910-161601-a45b8e`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260907-193948-30a684
- **Status:** done
- **Hypothesis / notes:** R11: first rough run whose feet actually touch the rocks (warp static-geom fix f8e7e27). Warm from the slope champion r5; task defaults: level-2 field covering the rough_v1 bar (24 boxes 6-15 mm), one-sided foot_clearance -2.0, stumble -0.5, feet_air_time 0.3, stall -0.05, 30M steps. Hypothesis: with real contacts the policy learns a high-step swing (>= 8 mm) and clears 0.45 m; bar_distance_x should track the benchmark, not sit 3x above it.
- **Config vs parent 20260907-193948-30a684:**
- `ppo.num_timesteps`: 10000000 → 30000000
- `reward.weights.feet_air_time`: 0.1 → 0.3
- `reward.weights.foot_clearance`: 0.0 → -2.0
- `reward.weights.stall`: 0.0 → -0.05
- `reward.weights.stumble`: 0.0 → -0.5
- `task`: slope_up → rough_walk
- `terrain.box_height_m`: [0.0, 0.0] → [0.006, 0.015]
- `terrain.box_share`: None → 1.0
- `terrain.box_size_m`: [0.02, 0.06] → [0.02, 0.05]
- `terrain.box_spacing_m`: 0.12 → 0.1
- `terrain.kind`: slope → rough
- `terrain.level`: 1 → 2
- `terrain.n_boxes`: 0 → 24
- `terrain.slope_deg`: [0.0, 8.0] → [0.0, 0.0]
- `terrain.slope_share`: None → 1.0
- `terrain.spawn_jitter_m`: 0.0 → 0.05
- **Training:** 30M steps, 2048 envs, 496 s, 60,429 steps/s (warp); eval reward 357 → 392
- **Gates:** 10/15; failing: forward_distance_p50 (0.393 >= 0.45), progress_ratio (0.328 >= 0.4), vel_tracking_rmse (0.111 <= 0.08), foot_clearance (0.659 >= 8.0), beats_baseline_trot_distance (1.033 >= 1.1)
- **Reflection:** BENCHMARK FAILED 10/15 gates (score 10.20). Fell in 0/20 episodes. Median distance 0.39 m in 10.0 s. Velocity tracking RMSE 0.111 m/s. FAIL forward_distance_p50: 0.393 >= 0.45 m (firmware trot 0.381) -> 1.1 x the firmware trot's 0.381 m across this field rounds to 0.42, kept at 0.45; ideal is 1.2 m at 0.12 m/s for 10 s FAIL progress_ratio: 0.328 >= 0.4 ratio (firmware trot 0.317) -> distance / (commanded speed x seconds); the robot is getting stuck on edges FAIL vel_tracking_rmse: 0.111 <= 0.08 m/s (firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel FAIL foot_clearance: 0.659 >= 8.0 mm (firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm) (+1 more failing gates in the table) Bar eval (the suite protocol's terrain and command, on the GPU, at the end of training): 0.35 m, falls 0% vs 0.39 m here: the engines agree; what the benchmark shows is what the policy learned. Parent 20260907-193948-30a684 has no rough_v1 benchmark (it was benchmarked on slope_v1), so there is no per-gate parent comparison. Re-benchmark it with bittle_benchmark_policy(run_id="20260907-193948-30a684", suite="rough_v1", force=true) if you want the delta.

### Run 17: r12-rough-swing-incentive-30M (`20260910-163154-8e82b7`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260910-161601-a45b8e
- **Status:** done
- **Hypothesis / notes:** R12: r11 learned to DRAG (swing clearance 0.66 mm, 0 stumbles, 0.39 m) even with foot_clearance at 17% share -- that term only scores swings, so never swinging is nearly free; the two terms that reward a real swing were invisible (feet_air_time 0.12%, feet_slip 0.02%). Hypothesis: feet_air_time 2.0 (20x) + feet_slip -2.0 (40x) make dragging expensive and swinging profitable; expect foot_clearance_p50 >= 8 mm and distance > 0.45 m.
- **Config vs parent 20260910-161601-a45b8e:**
- `reward.weights.feet_air_time`: 0.3 → 2.0
- `reward.weights.feet_slip`: -0.05 → -2.0
- **Training:** 30M steps, 2048 envs, 510 s, 58,863 steps/s (warp); eval reward 425 → 412
- **Gates:** 13/15; failing: vel_tracking_rmse (0.097 <= 0.08), foot_clearance (0.483 >= 8.0)
- **Reflection:** BENCHMARK FAILED 13/15 gates (score 13.27). Fell in 0/20 episodes. Median distance 0.54 m in 10.0 s. Velocity tracking RMSE 0.097 m/s. FAIL vel_tracking_rmse: 0.097 <= 0.08 m/s (parent 0.111, firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel FAIL foot_clearance: 0.483 >= 8.0 mm (parent 0.659, firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm). The 'foot_clearance' term already carries 14.8% of the reward yet the swing stays under 2 mm: the policy is DRAGGING its feet instead of swinging them, and a term that only scores swings cannot pay for a swing that never happens. The terms that make a real swing profitable are invisible (feet_air_time 0.93%, feet_slip 0.68%): raise reward.weights.feet_air_time and reward.weights.feet_slip 10-40x, not foot_clearance again. Bar eval (the suite protocol's terrain and command, on the GPU, at the end of training): 0.43 m, falls 0% vs 0.54 m here: the engines agree; what the benchmark shows is what the policy learned.

### Run 18: r13-rough-swing-bound-30M (`20260910-171039-9a4d1b`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260910-163154-8e82b7
- **Status:** done
- **Hypothesis / notes:** Hypothesis: r12's swing incentives (air_time 6.7x, slip 40x) bought speed but not clearance because both terms still carry <1% reward share and cannot steer. Push feet_air_time and feet_slip to the |10| bound so real high swings out-compete dragging (fix foot_clearance 0.48 -> >=8mm), and raise tracking_lin_vel 1.5->2.0 to close vel_tracking_rmse 0.097 -> <0.08. Watch stall_fraction (0.012) and peak_concurrent_stalls (2.0) which sit at threshold.
- **Config vs parent 20260910-163154-8e82b7:**
- `reward.weights.feet_air_time`: 2.0 → 10.0
- `reward.weights.feet_slip`: -2.0 → -10.0
- `reward.weights.tracking_lin_vel`: 1.5 → 2.0
- **Training:** 30M steps, 2048 envs, 478 s, 62,705 steps/s (warp); eval reward 608 → 585
- **Gates:** 13/15; failing: vel_tracking_rmse (0.092 <= 0.08), foot_clearance (0.567 >= 8.0)
- **Reflection:** BENCHMARK FAILED 13/15 gates (score 13.30). Fell in 0/20 episodes. Median distance 0.60 m in 10.0 s. Velocity tracking RMSE 0.092 m/s. FAIL vel_tracking_rmse: 0.092 <= 0.08 m/s (parent 0.097, firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel. The 'tracking_lin_vel' term already carries 49.7% of the reward and the gate did not move vs the parent: the weight is not the lever -- this needs a longer budget (ppo.num_timesteps 30M: a gait has to be reshaped, not tuned) or a harder training terrain, not another weight change. FAIL foot_clearance: 0.567 >= 8.0 mm (parent 0.483, firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm). The 'foot_clearance' term already carries 11.7% of the reward yet the swing stays under 2 mm: the policy is DRAGGING its feet instead of swinging them, and a term that only scores swings cannot pay for a swing that never happens. The terms that make a real swing profitable are invisible (feet_air_time 5.20%, feet_slip 2.59%): raise reward.weights.feet_air_time and reward.weights.feet_slip 10-40x, not foot_clearance again. Bar eval (the suite protocol's terrain and command, on the GPU, at the end of training): 0.41 m, falls 0% vs 0.60 m here: the engines agree; what the benchmark shows is what the policy learned.

### Run 19: r14-rough-budget-terrain-40M (`20260910-172154-94dafc`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260910-171039-9a4d1b
- **Status:** done
- **Hypothesis / notes:** Hypothesis: r13 moved both failing gates in the right direction while their reward terms now carry >=2% share (air_time 5.2%, slip 2.6%) -> this is a BUDGET problem, not a weight problem (reflection says the same). Extend to 40M steps to finish the drag->swing gait reshape, and raise training box heights 6-15mm -> 10-18mm so dragging on protocol-sized (12mm) rocks becomes unprofitable and ~60% of the field is at/above protocol height. Expect foot_clearance 0.57 -> >=8mm and RMSE 0.092 -> <=0.08.
- **Config vs parent 20260910-171039-9a4d1b:**
- `ppo.num_timesteps`: 30000000 → 40000000
- `reward.weights.feet_air_time`: 10.0 → 0.3
- `terrain.box_height_m`: [0.006, 0.015] → [0.01, 0.018]
- **Training:** 40M steps, 2048 envs, 640 s, 62,522 steps/s (warp); eval reward 524 → 505
- **Gates:** 11/15; failing: forward_distance_p50 (0.428 >= 0.45), progress_ratio (0.357 >= 0.4), vel_tracking_rmse (0.099 <= 0.08), foot_clearance (0.246 >= 8.0)
- **Reflection:** BENCHMARK FAILED 11/15 gates (score 11.16). Fell in 1/20 episodes. Median distance 0.43 m in 10.0 s. Velocity tracking RMSE 0.099 m/s. Falls happen on average at t=4.2 s. FAIL forward_distance_p50: 0.428 >= 0.45 m (parent 0.602, firmware trot 0.381) -> 1.1 x the firmware trot's 0.381 m across this field rounds to 0.42, kept at 0.45; ideal is 1.2 m at 0.12 m/s for 10 s FAIL progress_ratio: 0.357 >= 0.4 ratio (parent 0.502, firmware trot 0.317) -> distance / (commanded speed x seconds); the robot is getting stuck on edges FAIL vel_tracking_rmse: 0.099 <= 0.08 m/s (parent 0.092, firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel FAIL foot_clearance: 0.246 >= 8.0 mm (parent 0.567, firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm). The 'foot_clearance' term already carries 10.4% of the reward yet the swing stays under 2 mm: the policy is DRAGGING its feet instead of swinging them, and a term that only scores swings cannot pay for a swing that never happens. The terms that make a real swing profitable are invisible (feet_air_time 0.09%, feet_slip 2.46%): raise reward.weights.feet_air_time and reward.weights.feet_slip 10-40x, not foot_clearance again. Bar eval (the suite protocol's terrain and command, on the GPU, at the end of training): 0.35 m, falls 0% vs 0.43 m here: the engines agree; what the benchmark shows is what the policy learned. STUCK: 80% of episodes stalled (|vx| < 0.02 m/s for >= 0.5 s while commanded), median 3.3 s per episode (worst 8.5 s), first stall at x = 0.43 m; a shank/thigh was on the ground for 0% of the stalled steps -- the feet are NOT catching edges: it drags without lifting (foot_clearance / feet_air_time), or the tracking term is saturated (tracking_sigma).

### Run 20: r15-rough-budget-40M-pinned (`20260910-173522-0d6bc6`)

- **Task / suite:** rough_walk / rough_v1@0.2.0-uncalibrated; warm start from 20260910-171039-9a4d1b
- **Status:** done
- **Hypothesis / notes:** Hypothesis: r13 (30M, air_time 10 / slip -10 / tracking_lin_vel 2.0) moved both failing gates in the right direction with swing terms steering at >=2% share -> pure budget problem. Extend to 40M on the proven 6-15mm field to finish the drag->swing reshape. All reward keys explicitly pinned because the task stock patch silently reset feet_air_time to 0.3 in r14 and amputated the swing incentive (80% stalls, clearance 0.25mm). Expect foot_clearance 0.57 -> >=8mm and RMSE 0.092 -> <=0.08.
- **Config vs parent 20260910-171039-9a4d1b:**
- `ppo.num_timesteps`: 30000000 → 40000000
- **Training:** 40M steps, 2048 envs, 584 s, 68,556 steps/s (warp); eval reward 625 → 601
- **Gates:** 11/15; failing: vel_tracking_rmse (0.095 <= 0.08), foot_clearance (0.441 >= 8.0), stall_fraction (0.029 <= 0.02), peak_concurrent_stalls (4.000 <= 2)
- **Reflection:** BENCHMARK FAILED 11/15 gates (score 11.26). Fell in 0/20 episodes. Median distance 0.51 m in 10.0 s. Velocity tracking RMSE 0.095 m/s. FAIL vel_tracking_rmse: 0.095 <= 0.08 m/s (parent 0.092, firmware trot 0.113) -> tighten tracking_sigma or raise tracking_lin_vel. The 'tracking_lin_vel' term already carries 50.1% of the reward and the gate did not move vs the parent: the weight is not the lever -- this needs a longer budget (ppo.num_timesteps 30M: a gait has to be reshaped, not tuned) or a harder training terrain, not another weight change. FAIL foot_clearance: 0.441 >= 8.0 mm (parent 0.567, firmware trot 0.724) -> median swing-foot height above the local terrain; raise the foot_clearance penalty (target 12 mm). The 'foot_clearance' term already carries 11.1% of the reward yet the swing stays under 2 mm: the policy is DRAGGING its feet instead of swinging them, and a term that only scores swings cannot pay for a swing that never happens. The terms that make a real swing profitable are invisible (feet_air_time 5.62%, feet_slip 2.49%): raise reward.weights.feet_air_time and reward.weights.feet_slip 10-40x, not foot_clearance again. FAIL stall_fraction: 0.029 <= 0.02 fraction (parent 0.004, firmware trot 0.000) -> fraction of (control step, joint) pairs pinned at >= 90% torque cap while barely moving for the whole 20 ms step; a stalled P1S draws 1.5 A. Raise the stall penalty 10-100x or lower action_scale_deg. The 'stall' term is only 0.65% of the total reward, so the optimiser barely sees it: multiply its weight by ~3x (to ~2% share), not by 2-10x. FAIL peak_concurrent_stalls: 4.000 <= 2 count (parent 2.000, firmware trot 0.000) -> 3 simultaneously stalled servos draw 4.5 A, above the battery's 2 A continuous rating; 4 draw 6 A, above its 5 A peak. The 'stall' term is only 0.65% of the total reward, so the optimiser barely sees it: multiply its weight by ~3x (to ~2% share), not by 2-10x. Bar eval (the suite protocol's terrain and command, on the GPU, at the end of training): 0.47 m, falls 0% vs 0.51 m here: the engines agree; what the benchmark shows is what the policy learned.

## Runs outside the store (development smoke tests, 2026-09-07)

| run | steps | envs | result | lesson |
|---|---|---|---|---|
| smoke1 | 0.3M | 256 | reward 344 → 324, no learning | first end-to-end PPO run; too short to judge |
| crippled 40M (cancelled) | — | 2048 | cancelled after 80 s | servos could not move under the torque cap (kp 40, damping 1.5): trot replay 0.000 m |
| smoke2 | 3M | 1024 | reward 58 → 452, stands, falls when asked to walk (clip on the main page) | pipeline learns; 3M steps is ~6 episodes per env |

The first two ran on the original servo parameters, which is why they are not in the store: the model was rebuilt
with a torque-limited servo (kp 10, ±0.25 N·m, damping 0.05) before run 1.
