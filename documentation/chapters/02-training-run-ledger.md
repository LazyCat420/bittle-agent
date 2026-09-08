---
part: RL walking trainer
status: shipped
updated: 2026-09-07
---

# Training run ledger

Every policy training run recorded by the trainer, newest last. Generated from `runs/` by
`trainer/scripts/render_docs.py --ledger`; rerun it after training and commit the result.
Distances are the flat_v1 protocol (20 seeded episodes, 10 s, 0.12 m/s command, CPU MuJoCo on the
mesh model). Score = gates passed + 0.5·min(distance, 1) − fall rate.

## Runs in the store (3 runs)

| # | run | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20260907-135830-42943f`<br>r1-defaults-40M | 2026-09-07T20:58 | — | 40M | 2048 | 15.0 min | 15/15 | 1.13 m | 0.00 | 15.5 | [gif](media/rl-training/policy_r1-defaults-40M.gif) |
| 2 | `20260907-142153-d17a72`<br>r2-energy-fix-15M | 2026-09-07T21:21 | 20260907-135830-42943f | 15M | 2048 | 7.3 min | 15/15 | 1.19 m | 0.00 | 15.5 | [gif](media/rl-training/policy_r2-energy-fix-15M.gif) |
| 3 | `20260907-165840-e406b4`<br>r3-energy-x100-warm-10M | 2026-09-07T23:58 | 20260907-135830-42943f | 10M | 2048 | 6.4 min | 15/15 | 1.13 m | 0.00 | 15.5 | [gif](media/rl-training/policy_r3-energy-x100-warm-10M.gif) |

### Run 1: r1-defaults-40M (`20260907-135830-42943f`)

- **Status:** done
- **Hypothesis / notes:** first full run through the service (servo model kp10/0.25Nm)
- **Config vs defaults:**
- `init_from_parent`: True → None
- **Training:** 40M steps, 2048 envs, 900 s, 44,419 steps/s (warp); eval reward 16 → 762
- **Gates:** 15/15; failing: none
- **Reflection:** BENCHMARK PASSED 15/15 gates (score 15.50). Fell in 0/20 episodes. Median distance 1.13 m in 10.0 s. Velocity tracking RMSE 0.037 m/s. 4 gates not evaluated (stage/group/metric). All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

### Run 2: r2-energy-fix-15M (`20260907-142153-d17a72`)

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

- **Status:** done
- **Hypothesis / notes:** energy weight x100 (its share was 0.05% of reward); warm start from r1; 10M steps
- **Config vs parent 20260907-135830-42943f:**
- `init_from_parent`: None → True
- `ppo.num_timesteps`: 40000000 → 10000000
- `reward.weights.energy`: -0.0005 → -0.05
- **Training:** 10M steps, 2048 envs, 381 s, 26,217 steps/s (warp); eval reward 741 → 761
- **Gates:** 15/15; failing: none
- **Reflection:** BENCHMARK PASSED 15/15 gates (score 15.50). Fell in 0/20 episodes. Median distance 1.13 m in 10.0 s. Velocity tracking RMSE 0.028 m/s. 4 gates not evaluated (stage/group/metric). All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.

## Runs outside the store (development smoke tests, 2026-09-07)

| run | steps | envs | result | lesson |
|---|---|---|---|---|
| smoke1 | 0.3M | 256 | reward 344 → 324, no learning | first end-to-end PPO run; too short to judge |
| crippled 40M (cancelled) | — | 2048 | cancelled after 80 s | servos could not move under the torque cap (kp 40, damping 1.5): trot replay 0.000 m |
| smoke2 | 3M | 1024 | reward 58 → 452, stands, falls when asked to walk (clip on the main page) | pipeline learns; 3M steps is ~6 episodes per env |

The first two ran on the original servo parameters, which is why they are not in the store: the model was rebuilt
with a torque-limited servo (kp 10, ±0.25 N·m, damping 0.05) before run 1.
