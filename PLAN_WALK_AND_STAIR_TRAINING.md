# Plan: Bittle Scene Locomotion & Autonomous Stair-Climbing Training with GLM

## Executive Summary & Direct Answers to Questions

### 1. Can Bittle walk around the scene when GLM makes movements to climb the stairs?
**Yes.** Currently, in `static/js/viewer.js`, joint servos are animated in-place like a treadmill; `this.robotGroup.position.x` stays fixed at `0`, so Bittle never translates forward to reach the stairs ($x \ge 0.14\text{m}$). By adding **Kinematic Root Motion** and pitch conformance:
- When GLM commands movements (walking, stepping, trotting), Bittle will physically translate forward along the +X axis.
- As Bittle reaches $x = 0.14\text{m}$, it encounters the 3-step mini stairs.
- As the paws step onto the 18mm risers, Bittle pitches up realistically and climbs up to the top landing platform ($x \ge 0.38\text{m}$).

### 2. How much VRAM would it take to train?
- **GLM-Guided Simulation-in-the-Loop Training (Eureka / DrEureka Style): 0 GB additional VRAM!**
  - GLM (GLM-5.3-Flash or GLM-4) is already loaded and running on your local Gold Spark / Jetson cluster (`10.0.0.141:8000/v1`). It only generates text/JSON tool calls to propose and refine gait parameters.
  - The physics/kinematics simulation runs on CPU.
  - Zero model weights are being updated via gradient descent; GLM acts as the intelligent coach/evaluator.
- **Reinforcement Learning Policy Training (PPO MLP in PyBullet): 0 GB VRAM** (runs on CPU) or **~2 GB VRAM** if vectorized on GPU (Isaac Gym / Isaac Lab).
- *(Note: Attempting to fine-tune GLM's own LLM weights with LoRA would take 12–24 GB VRAM, but this is an anti-pattern because an LLM's 300–800ms latency cannot run a 50Hz dynamic balancing loop directly).*

### 3. Can we train with CPU?
**Yes, 100%.**
- Quadruped rigid-body physics engines (PyBullet or kinematic clearance evaluators) are lightweight and specifically designed to run on CPU.
- An 8-core CPU can execute PyBullet at **2,000 to 5,000 physics steps per second** (50x faster than real-time).
- A GLM-guided trial-and-error gait optimizer or a Stable-Baselines3 PPO RL training run can execute completely on CPU without needing any GPU resources.

### 4. How long would it take?
- **Approach A (Recommended: GLM Automated Gait Optimization Loop)**:
  - **~2 to 4 minutes total** (15 to 30 iterations).
  - Each run takes ~3–4 seconds: 2.5s GLM thought/parameter synthesis + 0.3s simulation rollout + 0.1s clearance evaluation.
- **Approach B (PyBullet PPO RL Policy on CPU)**:
  - **~20 to 45 minutes on CPU** for 1–2 million environment steps.
- **Approach C (Fine-tuning LLM weights with LoRA)**:
  - 2 to 6 hours on GPU, weeks on CPU (impractical and suboptimal).

---

## Technical Analysis: Why Bittle Walks In-Place Currently

1. In `static/js/viewer.js`, `setRobotPosition(x, z)` exists at line 208, but is never called during sequence or skill playback.
2. When GLM executes `bittle_do_skill("walk")` or `bittle_execute_sequence(...)`, `viewer.js` updates `targetAngles` and slews `currentAngles`. The chassis group `this.robotGroup.position.x` remains at `0`.
3. In `obstacle_course.js`, the 3-step `mini_stairs` begin at $x = 0.140\text{m}$:
   - **Step 1**: $x \in [0.140, 0.220]\text{m}$, elevation $y = 0.018\text{m}$ (18mm)
   - **Step 2**: $x \in [0.220, 0.300]\text{m}$, elevation $y = 0.036\text{m}$ (36mm)
   - **Step 3**: $x \in [0.300, 0.380]\text{m}$, elevation $y = 0.054\text{m}$ (54mm)
   - **Landing Platform**: $x \in [0.380, 0.520]\text{m}$, elevation $y = 0.054\text{m}$ (54mm)
4. While `viewer.js` has dynamic elevation tracking (`getElevationAt(robotGroup.position.x, robotGroup.position.z)`), because the robot never moves past $x = 0$, it always samples flat ground ($y = 0$).

---

## Architectural Breakdown: 2 Phases

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Scene Locomotion & Root Motion Kinematics                     │
│ • Assign per-step displacement & forward velocity to gaits             │
│ • Three.js root translation along +X (forward) & heading yaw           │
│ • Pitch calculation from front vs rear paw ground contact              │
│ • Step riser collision detection (trips if foot lift < 18mm)           │
│ • GLM Navigation tools: get_scene_pose, navigate_step, climb_step      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: GLM-in-the-Loop "Training Run" Engine (Zero Extra VRAM)       │
│ • GLM acts as the Curriculum Designer & Gait Optimizer (Eureka style)  │
│ • Automated trial runs against the 18mm mini-stairs                    │
│ • Simulation rollout evaluates: foot clearance, pitch, slip, success   │
│ • Telemetry feedback to GLM: "Tripped on step 2 at x=0.22m, lift 14mm" │
│ • GLM adjusts knee flexion / duty cycle / pitch bias and retries       │
│ • Commits winning moveset to library once full climb is verified       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Changes

### Component 1: 3D Twin Locomotion & Pitch Adaptation (`static/js/`)

#### [`viewer.js`](file:///home/lazycat/github/projects/sun/bittle-agent/static/js/viewer.js)
- **Kinematic Root Motion**:
  - Add `rootMotionEnabled` flag (default `true`).
  - Calculate forward velocity and step displacement from active sequences (e.g., `walk` / `trot` / `step_forward` / `stair_step_up`).
  - Translate `this.robotGroup.position.x` and `z` based on current heading $\psi$ during playback.
- **Dynamic Pitch Conformation**:
  - Sample front paw ground elevation $y_f$ and rear paw ground elevation $y_r$ using `this.obstacleCourse.getElevationAt(...)`.
  - Apply torso pitch rotation $\theta_{\text{pitch}} = \arctan\left(\frac{y_f - y_r}{L_{\text{wheelbase}}}\right)$ so Bittle's body angles up when climbing stairs.
- **Step Collision Detection**:
  - If robot attempts to cross a step riser ($x = 0.140, 0.220, 0.300$) while front knee flexion is insufficient (foot height $< \text{riserHeight}$), trigger a trip event: forward translation halts, robot pitches forward slightly, and a warning is emitted.
- **Camera Tracking**:
  - Camera smooth-follows the robot as it walks down the course.

#### [`obstacle_course.js`](file:///home/lazycat/github/projects/sun/bittle-agent/static/js/obstacle_course.js)
- Enhance `getElevationAt(x, z)` to return both elevation and surface normal for smooth pitch/roll alignment.
- Add `checkFootClearance(footX, footY, footZ)` for exact paw-riser collision checks.

---

### Component 2: Motion Primitives & Step Displacements (`app/motion/`)

#### [`primitives.py`](file:///home/lazycat/github/projects/sun/bittle-agent/app/motion/primitives.py) & [`builtin_library.py`](file:///home/lazycat/github/projects/sun/bittle-agent/app/motion/builtin_library.py)
- Annotate motion primitives and built-in movesets with kinematic displacement metadata:
  - `walk`: $+0.035\text{m}$ per stride along X
  - `trot`: $+0.045\text{m}$ per stride along X
  - `step_forward`: $+0.030\text{m}$ along X
  - `stair_step_up`: $+0.040\text{m}$ along X, $+0.018\text{m}$ along Y
  - `crawl`: $+0.020\text{m}$ along X
  - `turn_left` / `turn_right`: $\pm 15^\circ$ yaw

#### [`obstacles.py`](file:///home/lazycat/github/projects/sun/bittle-agent/app/motion/obstacles.py)
- Add stair trajectory evaluator: validates multi-step sequences across all 3 steps $(x = 0.14 \to 0.40\text{m})$.
- Provide detailed failure feedback strings designed for LLM reflection (e.g. exact step index, riser collision coordinate, deficit in mm).

---

### Component 3: GLM Agent Tools & Training Run Harness (`app/agent.py`)

#### [`agent.py`](file:///home/lazycat/github/projects/sun/bittle-agent/app/agent.py)
- Add new tools for GLM:
  1. `bittle_get_scene_pose`: Returns robot $(x, y, z)$, heading, active course, and distance to the next obstacle/riser.
  2. `bittle_navigate_to`: Navigates Bittle to a target $(x, z)$ waypoint using step primitives.
  3. `bittle_run_stair_training_episode`: Executes an automated trial run. GLM passes candidate gait parameters or moveset; the harness simulates the climb, evaluates foot clearance and pitch stability, and returns structured telemetry.
- Add autonomous training loop:
  - GLM can run an iterative session:
    ```
    Loop:
      1. Propose/Adjust gait parameters (knee lift, timing, torso pitch bias)
      2. Run episode on mini_stairs
      3. If success: Save verified moveset 'stair_climb_v1', report victory
      4. If trip/fail: Reflect on telemetry (e.g. "Front right paw hit riser 2"), adjust, repeat (max N runs)
    ```

---

### Component 4: Web UI & Timeline (`static/js/agent-ui.js` & `static/index.html`)

#### [`agent-ui.js`](file:///home/lazycat/github/projects/sun/bittle-agent/static/js/agent-ui.js)
- When `bittle_navigate_to` or `bittle_run_stair_training_episode` executes, stream live progress to the UI.
- Display robot coordinates $(x, y, z)$ on the 3D HUD overlay.
- Add "Reset Position" button in the 3D viewer toolbar.

---

## Verification Plan

### Automated Tests
1. **Kinematic Root Motion Test**:
   - `pytest tests/test_root_motion.py`: Verify that executing `step_forward` and `walk` accurately increments the robot's pose coordinates $(x, z)$.
2. **Stair Traversal & Clearance Test**:
   - `pytest tests/test_stair_traversal.py`: Test that standard walking correctly detects riser collisions on Step 1, while `stair_step_up` successfully clears all 3 steps onto the landing platform.
3. **Training Run Episode Harness Test**:
   - `pytest tests/test_training_episode.py`: Mock a multi-iteration tuning loop and verify that parameter adjustments produce valid movesets and telemetry.

### Manual Verification
1. Open the Bittle Web UI (`http://localhost:8008` or NAS container `:8008`).
2. Select **Terrain: Mini Stairs**.
3. Command GLM via chat: *"Walk forward towards the stairs and climb them."*
4. Confirm Bittle visibly walks forward from $(0, 0)$, reaches the stairs at $x = 0.14\text{m}$, lifts paws onto Step 1, tilts realistically, climbs onto Step 2 and Step 3, and stands on the top landing platform.
5. Trigger a training run: *"Run training iterations to optimize stair climbing."* Observe GLM iteratively refining the gait and announcing the result.
