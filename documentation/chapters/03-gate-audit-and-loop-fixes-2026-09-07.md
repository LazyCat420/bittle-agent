---
part: RL walking trainer
status: shipped
updated: 2026-09-07
---

# The gate that failed was the gate, not the robot (audit + loop fixes)

## The question

Runs 1 and 2 both walk (1.13 m and 1.19 m in 10 s, no falls, faster than the firmware trot), yet both showed
"13/14 gates, energy_proxy FAILED". Is training working or not? **It is working. The one failing gate was
miscalibrated, and the harness was giving the LLM feedback it could not act on.** Both are fixed.

## Finding 1: the energy threshold was a guess, and the plan said so

`energy_proxy` is the mean of |torque × joint speed| over all eight servos, in watts. The threshold of
1.0 W was written into the plan as an "initial threshold, calibrated once against the trot baseline before
freezing". That calibration step was skipped. Measured on the same protocol:

| gait | energy proxy | distance / 10 s |
|---|---|---|
| firmware trot (`opencat_trF`, replayed) | **3.79 W** | 0.997 m |
| firmware walk (`opencat_wkF`) | 2.79 W | 0.745 m |
| policy r1 | 1.08 W | 1.13 m |
| policy r2 | 1.39 W | 1.19 m |

The learned policies already use a quarter to a third of the firmware trot's power while walking further.
A gate that fails them is not measuring a defect.

**Fix:** gate suite `flat_v1` bumped to **1.1.0**: `energy_proxy ≤ 2.0 W` (absolute sanity cap) and a new
`energy_vs_baseline ≤ 0.5` (a learned gait must use at most half the trot's power). Re-benchmarked under
1.1.0, **r1 and r2 pass 15/15 evaluated gates**. The leaderboard only ranks same-version reports, so both
were re-run; the 1.0.0 reports are kept on disk.

## Finding 2: the LLM was told "raise the energy penalty" but not by how much

Reward is a per-step weighted sum. In run 1 the energy term contributed **−0.4 of a 762-point episode
reward (0.05 %)**; velocity tracking contributed 580. A term that small is invisible to the optimiser, so
GLM's ×10 change (still 0.5 %) could not move behaviour, and its second change (foot-air-time ×2.5) is what
actually changed the gait. GLM diagnosed that correctly afterwards, but the harness should have said it first.

**Fix:** every gate report now carries **context** — for each gate the parent run's value and the firmware
trot's value, plus the reward-term breakdown as a share of the total. The reflection text uses it:

> FAIL energy_proxy: 2.500 <= 2.0 W (parent 1.080, firmware trot 3.790) → … The 'energy' term is only
> 0.07 % of the total reward, so the optimiser barely sees it: multiply its weight by ~30x (to ~2 % share),
> not by 2-10x.

A new tool, `bittle_diagnose_run`, returns the same table on demand, and the training protocol tells GLM to
call it before choosing a patch and to move weak terms by 10–100×.

## Finding 3: every cycle re-trained from scratch

A 15M-step child run started from a random policy and spent most of its budget re-learning to walk. Child
runs now **warm-start from the parent's policy and observation normaliser** (`init_from_parent`, default on,
same network shape required), and the protocol asks for 10M-step cycles. Measured on run 3 below.

## Finding 4: a flaky service test was a real race

`RunStore.update_state` is a read-modify-write on `state.json`; the job loop, the fake-job threads and the
request handlers all call it, and one lost update made the contract test fail under CPU contention. It is
now serialised with a lock, and the reaper's "benchmark exited without a report" rule only applies to
benchmark processes.

## Run 3: the fixes in one cycle

`r3-energy-x100-warm-10M`: base = run 1, energy weight −0.0005 → −0.05 (×100, the size the reward share
called for), 10M steps, warm-started.

| | run 1 (from scratch, 40M) | run 3 (warm start, 10M) |
|---|---|---|
| first eval reward | 16 | **741** (the parent's policy, before any training) |
| wall-clock | 15 min | **6 min** |
| energy proxy | 1.08 W | **0.95 W** (0.25× the firmware trot) |
| energy term's reward share | 0.05 % | 3.6 % |
| distance / 10 s | 1.13 m | 1.13 m |
| velocity tracking RMSE | 0.037 m/s | **0.028 m/s** |
| gates (1.1.0) | 15 / 15 | 15 / 15 |

One cycle, one hypothesis, a third of the time, and the metric the weight targets moved while nothing else
regressed. That is what a working loop looks like: the curve is flat because the run starts converged; the
gate table is where the change shows.

![run 3](media/rl-training/policy_r3-energy-x100-warm-10M.gif)

## How to read the results page from now on

- **Curves** (reward, distance during training) answer "is this run still learning". They do not compare runs:
  each run's reward is scored by its own weights.
- **Gate table** on the fixed CPU protocol compares runs. Same seeds, same model, same command.
- **Reward shares** (`bittle_diagnose_run`) tell you which weight is worth touching.
- The next lever after flat-ground gates is not harder terrain yet: run the **DR sweep** (`dr_sweep`) to check
  robustness to friction, payload, servo strength and latency, then **curriculum stage 1** (turning, 0–0.25 m/s),
  then stage 2 (lateral, pushes). Terrain (the 18 mm mini-stairs from the obstacle course, as box geoms) comes
  after that — it is a new gate suite, not a tweak. *(Done the same day: chapter 04 adds the incline and
  rock suites, the servo-safety gates and the task catalogue; the order above still holds for a new parent.)*
