# bittle-trainer

RL locomotion training for the Petoi Bittle X, driven by the GLM agent in
bittle-agent. Runs on a GPU box (the WSL2 dev PC with the RTX 3090 Ti), **not**
in the NAS container. The agent never writes training code: it edits a
validated JSON config, launches a run, reads a gate report, and iterates.

```
GLM (Gold Spark) ──tools──▶ bittle-agent (NAS :8008) ──HTTP──▶ trainer (GPU box :8009)
                                                               ├─ train:  MuJoCo Warp + Brax PPO (GPU)
                                                               ├─ eval:   CPU MuJoCo, full-mesh model, seeded
                                                               ├─ gates:  eval/suites/<suite>.yaml = the policy test suites (per task)
                                                               └─ runs/:  configs, policies, reports, rollouts
```

## Setup (GPU box)

```bash
trainer/scripts/setup_venv.sh          # .venv-trainer with pinned deps (warp-lang MUST be 1.16.0)
.venv-trainer/bin/python -m trainer.assets.build_models   # regenerates trainer/assets/generated/*.xml
trainer/scripts/run_service.sh         # :8009, 8 cores, nice 10, half the VRAM
.venv-trainer/bin/python -m trainer.eval.baselines        # once: OpenCat gait baselines -> runs/baselines/
```

Then point bittle-agent at it: `BITTLE_TRAINER_URL=http://<pc-ip>:8009` (from the
NAS, WSL2 needs mirrored networking or a `netsh portproxy`). Open the control
panel, tick **Training mode**, and ask GLM to train.

Docker (secondary): `docker compose -f trainer/docker-compose.yml up --build`
(needs the NVIDIA Container Toolkit in the WSL dockerd).

## What the agent sees

| tool | does |
|---|---|
| `bittle_propose_config` | validate a config patch, see the diff and warnings |
| `bittle_list_tasks` | the task catalogue (flat_walk, slope_up, rough_walk): goal, suite, prerequisite met?, warm-start run |
| `bittle_train_and_benchmark` | ONE cycle: train a task → wait → that task's gate suite → report + reflection |
| `bittle_train_policy` / `bittle_train_status` / `bittle_benchmark_policy` | the same, step by step |
| `bittle_list_runs` / `bittle_compare_runs` | leaderboard, baselines, gate tables, config diffs |
| `bittle_replay_rollout` | play a benchmark rollout (or `baseline:opencat_trF`) in the 3D viewer |
| `bittle_web_search` / `bittle_search_papers` / `bittle_read_url` / research notes | look things up, remember findings |

The config the agent edits is `trainer/config.py::TrainConfig` — bounded,
`extra="forbid"`, hashed. The gates are `trainer/eval/suites/<suite>.yaml` (every suite includes the
`_servo_safety` fragment; a run's task decides its suite, see `trainer/tasks.py`); changing a
threshold bumps the suite version and the leaderboard only ranks same-version reports.

## Model

`trainer/assets/build_models.py` turns `static/assets/bittle.xml` (measured
masses, 273.5 g) into two variants: `bittle_cpu.xml` (mesh collisions, used
for evaluation) and `bittle_gpu.xml` (primitive collisions, used for batched
training). Both weld the unactuated neck, add foot spheres + contact sensors,
and use a torque-limited servo model (kp 10 N·m/rad, ±0.25 N·m, damping 0.05).
The joint conventions (rear-shoulder mirror, 85° vs 80° knee) live only in
`trainer/assets/joint_map.py`.

## Observation / action

41-dim observation: gravity vector, gyro, velocity command, the last 3
commanded joint targets, last action — everything a BiBoard can provide from
the IMU plus its own command history (no joint encoders). Actions are
per-step target deltas (6°/step), clipped to the hardware profile's agent
envelope, rounded to integer degrees, delayed by 0–3 steps (DR).

## Reproduce without the LLM

```bash
python -m trainer.campaign --agent-url http://nas:8008 --strategy scripted_v1   # through bittle-agent tools
python -m trainer.campaign --in-process --trainer-url http://127.0.0.1:8009       # direct
```

## Tests

```bash
.venv-trainer/bin/python -m pytest trainer/tests -q                 # CPU: config, gates, service, models, env
TRAINER_GPU_TESTS=1 .venv-trainer/bin/python -m pytest trainer/tests -q -m gpu   # + Warp/JAX parity, export
.venv-trainer/bin/python -m trainer.train.smoke --num-envs 1024 --num-timesteps 3000000   # learning smoke
```

`deploy.sh` runs only the root `tests/` (the NAS gate); `trainer/tests` never runs on the NAS.
