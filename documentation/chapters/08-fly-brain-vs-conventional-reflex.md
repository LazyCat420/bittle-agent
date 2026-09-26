---
part: RL walking trainer
status: in-progress
updated: 2026-09-26
review-by: 2026-10-10
---

# Fly brain vs. conventional reflex: the four arms are built, and the pilot cannot tell them apart

**Filed 2026-09-26 by the daily documentation review.** The plan behind the
repo's newest commits lived only in `PLAN_FLY_BRAIN_AB_BALANCE_ROCKS_AND_BALL.md`
at the repo root — a tree the review does not read — so the work it governs has
been undocumented since **2026-09-12**, which is also the last day this repo was
committed to. This chapter files it, and adds the thing the plan cannot contain:
**what the pilot returned.**

## The question and the four arms

Does circuitry derived from the *Drosophila* connectome beat classical control
at quadruped stabilisation? The plan's own framing is that the obvious
experiment is confounded — *"it conflated the benefit of adding an auxiliary
stabilizer with the benefit of the fly connectome itself"* — so it runs four
arms, and the fourth is the one that makes it an experiment:

| arm | controller | what it isolates |
|---|---|---|
| **A** `none` | PPO alone | the floor |
| **B** `conventional` | PPO + PD attitude reflex | the benefit of *any* auxiliary stabiliser |
| **C** `fly` | PPO + engineered fly-derived recurrent circuit | the hypothesis |
| **D** `null` | PPO + degree- and weight-matched **rewired** circuit | the benefit of *that specific wiring* |

Arm D is the control that makes C falsifiable: same connectivity statistics,
shuffled topology. If C does not beat D, the connectome contributed nothing
its degree distribution did not.

Two suites: `rough_walk` (the existing rock terrain) and `ball_balance` — Bittle
standing on a free-rolling size-7 basketball, thin hollow shell inertia
`I = ⅔mr² = 0.00576 kg·m²`, `condim=6` with torsional and rolling friction.

## Stage 1 shipped

All of it is on `main`, measured today:

| piece | where |
|---|---|
| four reflex controllers + the rewired null | `trainer/policy/null_reflex.py`, `trainer/tests/test_reflex_controllers.py` |
| `ReflexConfig` wired into training and benchmarking | `trainer/config.py`, `benchmark_run` (`da6b247`) |
| `ball_balance` task + `ball_v1` suite | `trainer/tasks.py`, `trainer/eval/suites/ball_v1.yaml` (`b3c5b21`) |
| basketball model auto-routed for the task | `trainer/env/gpu_env.py`, `trainer/env/cpu_env.py` (`061dbf5`) |
| physics fixes the ball forced | `support_h` + contact budget (`9af3d1b`), centred spawn pose (`1cfe4cf`) |

The contact-budget fix is the same defect [ch.07](07-stairs-and-rubble.md)
records for dense rubble, arriving a third time through a new workload.

## ⚠ Stage 2 ran, and its primary metric is saturated

Five `ball_balance` runs on 2026-09-12 — the four arms at 3M steps, then one
`fly` arm at 12M. Read from each run's own
`benchmark/ball_v1@1.0.0-uncalibrated/report.json`:

| arm | steps | **fall rate** (gate ≤ 0.20) | gates passed | reward |
|---|---|---|---|---|
| A `none` | 3M | **1.00** | 8/10 | 68.5 |
| B `conventional` | 3M | **1.00** | 7/10 | 58.0 |
| C `fly` | 3M | **1.00** | 8/10 | 65.1 |
| D `null` | 3M | **1.00** | 8/10 | 74.4 |
| C `fly` | 12M | **0.95** | 9/10 | 391.9 |

**Every arm falls off the ball in every episode.** The primary outcome metric —
the one C-05, C-06 and C-07 are all stated in terms of — is pinned at its
ceiling for all four arms, so it separates nothing. The comparison is not
*close*; it is **undefined**.

Two things follow, and both are traps the plan's own VCPM discipline should
catch before Stage 3:

- **The secondary gates invert the hypothesis.** Ranked by gates passed, the
  **rewired null control (D, 8/10) beats the conventional reflex (B, 7/10)**,
  and by reward D is first (74.4) and B is last (58.0). Since no arm survives
  a single episode, that ordering is measuring vertical-oscillation and
  servo-safety noise, not stabilisation. **Do not read it as a result** — it is
  the shape where a shuffled control "wins" because the outcome metric is dead.
- **More training moves it, but not enough to test anything.** 4× the steps
  buys fall rate 1.00 → 0.95, episode length 98 → 252 steps, and final
  x-drift −0.15 m → −0.017 m. The direction is right and the gate is still
  failed, so 30M steps may or may not clear it — that is now the open question,
  not a safe assumption.

> **A gate every arm fails at the ceiling cannot compare the arms.** Before
> spending a 5-seed × 30M-step Stage 3, get *one* arm — any arm — to a fall
> rate the gate can actually resolve. Until then the experiment has no
> measuring instrument, and five seeds of an unresolvable comparison costs
> compute and returns a number that means nothing.

## Status per claim

- **C-01 – C-04** (verified facts: actuation limits, terrain geometry, shell
  inertia, EB/PB 1-DoF encoding) — carried over from the plan; not re-verified
  by this review.
- **C-05 / C-06 / C-07** (the three testable claims) — **untestable as
  measured.** Not refuted: no arm was given a condition in which it could
  differ. This is the distinction the plan exists to protect.

## What is left

1. Raise `ball_balance` out of saturation — longer training, an easier rung
   (a larger or partly damped ball, a shorter horizon), or a curriculum. The
   curb-curriculum result in [ch.07](07-stairs-and-rubble.md) §s2 is the local
   precedent: an easier first ask bought the behaviour the hard ask could not.
2. Only then Stage 3 (5 seeds × 30M, bootstrap 95% CIs).
3. The `rough_walk` suite arm of the A/B/C/D comparison does not appear to have
   been run at all — no run under `runs/` carries it. Confirm before assuming.
4. `ball_v1` is `1.0.0-uncalibrated`. Per ch.07's standing rule, freeze it only
   once a policy sets a bar worth keeping; and per that chapter's own hard-won
   lesson, check the bars are not sitting at the task's geometric ceiling.

The root plan file is left in place as the detailed protocol — the VCPM table,
the actuation model and the physics specification are worth reading in full and
are not reproduced here.
