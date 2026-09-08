---
part: RL walking trainer
status: shipped
updated: 2026-09-07
---

# Teaching Bittle to walk: the RL training loop

*Status 2026-09-07. Everything below was measured on the WSL2 dev PC (RTX 3090 Ti) unless marked otherwise.*

Yes — the walking training has run. Two policies exist, both walk, and the GLM agent drove the second one
itself. This page is the reference for how the loop works, what the numbers mean, and how to run it.

| Petoi trot (firmware gait, replayed) | Policy r1, 40M steps (defaults) | Policy r2, GLM's own cycle |
|---|---|---|
| ![](media/rl-training/baseline_opencat_trot.gif) | ![](media/rl-training/policy_r1-defaults-40M.gif) | ![](media/rl-training/policy_r2-energy-fix-15M.gif) |

All three clips: same simulator, same seed, same command (walk forward at 0.12 m/s), 5 s. The caption on each
frame shows the distance covered. The neural-net policies walk further than the hand-tuned firmware trot.

---

## 1. What was built

```
GLM-5.3-Flash (Gold Spark vLLM)
      │ tool calls
      ▼
bittle-agent  (NAS :8008, 512 MB, no GPU)  ──HTTP──▶  bittle-trainer  (WSL2 PC :8009, RTX 3090 Ti)
      ▲                                                  ├─ train/  MuJoCo Warp + Brax PPO, 2048 envs on the GPU
      │ gate table + reflection                          ├─ eval/   CPU MuJoCo, full-mesh model, 20 seeded episodes
      │                                                  └─ gates.yaml = the policy test suite
      └── viewer_url ──▶ 3D viewer rollout playback
```

The agent never writes training code. It edits a **validated JSON config** (reward weights, curriculum stage,
domain-randomisation ranges, PPO hyper-parameters), launches a run, reads a **gate report** (pass/fail per
metric plus a plain-English reflection), and proposes the next patch. One tool call,
`bittle_train_and_benchmark`, is one full cycle.

Code map (all under `bittle-agent/`):

| path | role |
|---|---|
| `trainer/config.py` | `TrainConfig` — the only thing the LLM edits; bounded, unknown keys rejected, hashed |
| `trainer/assets/build_models.py` | generates the two MJCF variants from `static/assets/bittle.xml` |
| `trainer/assets/joint_map.py` | the single place the joint conventions live (rear-shoulder mirror, 85° vs 80° knee) |
| `trainer/env/spec.py` | observation / action / reward / termination, shared by CPU and GPU envs |
| `trainer/env/gpu_env.py`, `cpu_env.py` | MuJoCo Warp (or MJX) batched env; CPU evaluator env |
| `trainer/train/ppo.py` | Brax PPO wrapper, policy export to `policy.npz` |
| `trainer/eval/gates.yaml`, `gates.py` | the gate suite `flat_v1@1.0.0` and its reflection text |
| `trainer/eval/evaluator.py`, `baselines.py` | seeded episodes, OpenCat gait replay, rollout recorder |
| `trainer/service.py` | FastAPI on :8009, async jobs with long-poll |
| `trainer/campaign/` | scripted search through the same tools, no LLM (`campaign.json` is the reproduction log) |
| `app/training_tools.py`, `app/research_tools.py` | the GLM tools; `app/trainer_client.py` the HTTP client |

---

## 2. The simulated robot

`static/assets/bittle.xml` is a MuJoCo model with **physically weighed masses** (273.5 g total, 55 g battery,
10.7 g per servo). Nothing in the repo loaded it before; the trainer now generates two variants from it.

| visual meshes | mesh collisions + foot spheres (CPU evaluator) | primitive collisions (GPU training) |
|---|---|---|
| ![](media/rl-training/model_cpu_visual_meshes.png) | ![](media/rl-training/model_cpu_mesh_collision_geoms_and_foot_spheres.png) | ![](media/rl-training/model_gpu_primitive_collision_model.png) |

Both variants weld the unactuated neck, add a 6 mm sphere at each paw tip with a **contact sensor**, and use
a torque-limited servo model (P1S-like: kp 10 N·m/rad, ±0.25 N·m, joint damping 0.05). They settle at the
same standing height (47.1 mm vs 47.0 mm), so a policy trained on the primitive model is graded on the mesh model.

> The source model used kp 40 with joint damping 1.5 and **no torque limit**. With a realistic torque cap those
> joints cannot move at all (max 7°/s): the firmware trot measured 0.000 m and the first PPO runs "learned to
> stand". Always check `measured_deg` moves before trusting a reward curve.

---

## 3. What the policy sees and does

**Observation (41 numbers)** — only things the real BiBoard can know from its IMU and its own command history:

| block | size |
|---|---|
| gravity direction in the body frame | 3 |
| gyro | 3 |
| velocity command (vx, vy, yaw rate) | 3 |
| last 3 commanded joint targets, relative to the stand pose | 24 |
| last action | 8 |

No joint encoders, no base velocity. **Action:** 8 target deltas of up to 6°/step at 50 Hz, clipped to the
hardware profile's *agent* envelope, rounded to integer degrees, delayed 0–3 steps by domain randomisation.

**Reward** (per step, weights are the LLM's knobs): velocity tracking (exp of the squared error, σ = 0.01
because Bittle moves at 0.1–0.2 m/s), yaw-rate tracking, and penalties for vertical bounce, roll/pitch rate,
tilt, height error, action rate, energy (|τ·q̇|), joint saturation, foot slip, and moving when told to stand;
a small reward for foot air time.

**Domain randomisation:** floor friction 0.5–1.1, mass ±15 % plus 0–30 g payload, servo kp 6–15, torque
0.15–0.35 N·m, joint damping and friction, control latency 0–3 steps, gyro noise and bias, initial pose noise;
pushes from curriculum stage 2.

---

## 4. Training results so far

![training curves](media/rl-training/training_curves.png)

| run | steps | wall-clock | throughput | gates | distance / 10 s | falls |
|---|---|---|---|---|---|---|
| smoke (1024 envs) | 3M | 137 s | 43k steps/s | — | stands, then falls under the walk command | — |
| **r1-defaults-40M** | 40M | **15 min** | ~50k steps/s | **15 / 15** (13/14 under gates 1.0.0) | **1.13 m** | 0 / 20 |
| **r2-energy-fix-15M** (GLM's patch) | 15M | 7.3 min | 34k steps/s | 15 / 15 (13/14 under 1.0.0) | 1.19 m | 0 / 20 |
| OpenCat `trF` baseline | — | — | — | 8 / 14 | 0.997 m | 0 / 20 |

![gates](media/rl-training/gates.png)

Under the original gate suite (1.0.0) both policies failed exactly one gate, `energy_proxy` (1.08 W and
1.39 W against a 1.0 W limit). The [audit](#the-gate-that-failed-was-the-gate-not-the-robot-audit-loop-fixes)
found the limit was an uncalibrated guess: the firmware trot itself measures 3.79 W on the same proxy. Under
the calibrated suite (1.1.0) both runs pass every evaluated gate. The gate chart below is the 1.0.0 report,
kept as the record of what the agent was reacting to. The GLM cycle raised the energy penalty ×10 **and**
raised the foot-air-time reward; the second change won, the gait got longer and faster, and energy went up.
GLM's own report says so and proposes reverting foot-air-time and raising the energy penalty next.

The learned gait is a trot — diagonal pairs land together:

![gait diagram](media/rl-training/gait_diagram.png)

Before training, for contrast (the 3M-step smoke policy falls as soon as it is asked to walk):

![smoke](media/rl-training/policy_smoke_3M_steps.gif)

---

## 5. The agent's loop, as it actually happened

Transcript: `runs/glm-session-2.sse` (SSE events from the local bittle-agent, training mode).

1. `bittle_trainer_health` → Warp backend, 3090 Ti, no jobs.
2. `bittle_list_runs` → leaderboard with r1 at 13/14 and the six baselines.
3. `bittle_propose_config` with `base_run_id` = r1 → the resolved config and editable keys.
4. `bittle_benchmark_policy` on r1 → re-read the failing gate.
5. `bittle_train_and_benchmark` with a 3-key patch and a written hypothesis; 30 progress keepalives over
   7.5 minutes; gate table and reflection returned.
6. Final report: which gates passed, an honest diagnosis of why energy got worse, the next hypothesis.

Every training-mode chat has a 12-turn budget and a 600 s LLM read timeout (the prompt carries ~37 tool
schemas and GLM queues behind other sequences on Gold Spark).

---

## 6. Running it

```bash
# GPU box (this PC), once
trainer/scripts/setup_venv.sh                 # pinned deps; warp-lang MUST be 1.16.0
.venv-trainer/bin/python -m trainer.assets.build_models
.venv-trainer/bin/python -m trainer.eval.baselines      # OpenCat gait baselines -> runs/baselines/

# GPU box, every session
trainer/scripts/run_service.sh                # :8009, 8 cores, nice 10, half the VRAM

# regenerate the figures on this page
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-trainer/bin/python trainer/scripts/render_docs.py
```

Then open the control panel, tick **Training mode**, and use the chips ("First training cycle",
"Train until gates pass", "Research then propose", "Leaderboard & replay").

Without the LLM: `python -m trainer.campaign --agent-url http://nas:8008 --strategy scripted_v1`.

### The one network step

`npm run deploy` pushes the **container to the NAS** — that direction works and has been done. The trainer
is the other direction: the container on the NAS must open connections **into this PC** on port 8009. Windows
blocks those inbound connections to the WSL2 address (10.0.0.171) until a rule exists. In an admin PowerShell:

```powershell
New-NetFirewallRule -DisplayName "bittle-trainer 8009" -Direction Inbound -Protocol TCP -LocalPort 8009 -Action Allow
```

Until then the deployed agent answers every training tool with `trainer_unavailable`. Running bittle-agent
locally on the PC (`npm run dev` with `BITTLE_TRAINER_URL=http://127.0.0.1:8009`) needs no rule, which is how
the GLM session above was run.

---

## 7. Sizing and box choice

| path | measured | notes |
|---|---|---|
| MuJoCo Warp, 4096 envs, raw physics | 521k physics steps/s | `bittle_gpu.xml` |
| full env (obs + reward + DR), Warp | 24k steps/s @512 envs, ~50k @2048 | PPO JIT ≈ 1 min |
| same env on MJX (JAX) | 4k steps/s @512 envs | fallback, one flag |
| CPU MuJoCo, one thread | 34k physics steps/s (mesh) | gates: 20 episodes ≈ 1 min |
| VRAM | a few GB; JAX capped at 50 % | |

The Sparks were not used on purpose: the job is tiny, they are aarch64 with sm_121 caveats, and they serve GLM.

---

## 8. Not done yet

- **Sim-to-real is out of scope.** Before any policy is driven onto the BiBoard, bittle-agent's joint-degree
  convention (STAND −45/80) must be translated to firmware degrees (OpenCat balance 30/30); the built-in
  keyframe trot/walk movesets also walk *backwards* in physics, so they were never physically validated.
- Curriculum stages 1–2 (turning, lateral, pushes), the DR-sweep gate and the GPU-vs-CPU consistency gate are
  implemented but have not been exercised by a real run.
- The 3D viewer rollout playback was checked on the JSON, not by eye.
