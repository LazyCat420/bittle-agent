# Plan: Bittle Obstacle Course, Terrain Adaptation & Hierarchical Agent Control

## Overview & Executive Summary

This plan addresses:
1. **Obstacle Courses & Stairs**: The physical and kinematic constraints of the Petoi Bittle quadruped, what kinds of courses/stairs it can physically climb, and how to build simulated & physical courses.
2. **Control Division of Labor & Latency**: Why having GLM directly run the balance/reflex loop will fail due to high latency (~300–1000ms vs. the required 10–30ms), and why a **3-tier Hierarchical Control Architecture** (Firmware Reflexes ⇄ Neural Locomotion Policy ⇄ GLM Strategic Choreographer) is the universal industry standard.
3. **How Far Can We Take This**: A roadmap from simulated 3D WebGL obstacle courses to PyBullet/Isaac Gym RL policy generation (DrEureka-style) to multi-modal Vision-Language-Action (VLA) navigation.

---

## 1. Deep Latency & Control Architecture Analysis

### Why Direct LLM Balance Control Sucks
- **The Physics of Quadruped Balance**:
  - Bittle operates at step frequencies of ~1.5–3.5 Hz. During each step cycle (300–600ms), single-foot touch-down, ground reaction forces, and slip events happen within **5 to 20 milliseconds**.
  - To prevent tripping on a stair edge or falling over, corrective joint angle adjustments must happen at **50 Hz to 200 Hz** (every 5ms to 20ms).
- **Latency Breakdown for Direct LLM Control**:
  - **Local LLM Inference** (GLM-4 / GLM-4-9B on DGX Spark / Jetson Orin): ~200ms – 800ms minimum per turn.
  - **HTTP & MCP Tool Call Roundtrip**: ~10ms – 25ms.
  - **OpenCat Serial Wire Latency**: 115,200 baud ASCII transmission + firmware echo ACK: ~25ms – 50ms.
  - **Total Loop Latency**: **300ms to 1000+ ms (1 to 3 Hz)**.
- **The Verdict**:
  > **If GLM is in the tight balance loop, the robot will tumble and crash to the floor 5 times over before GLM can finish tokenizing its response.**
  Therefore, **GLM should NOT be in the tight balance loop.**

---

### The Proven 3-Tier Hierarchical Architecture

To give Bittle autonomous balance and terrain adaptability while leveraging GLM's intelligence, we use the decoupled hierarchical model adopted by Boston Dynamics, MIT, and NVIDIA:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: Cognitive Strategy & Choreographer (GLM-4 / GLM-4-9B)              │
│  Rate: 0.2 – 1 Hz (or Event-Driven)                                         │
│  • Scene understanding: "I see 20mm stairs ahead."                          │
│  • High-level mission: Sets target velocity [vx, vy, yaw_rate] & gait style │
│  • Novel Moveset Synthesis: Composes custom stepping/recovery sequences     │
│  • DrEureka / Eureka Loop: Generates & iterates reward/trajectory code      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ High-level goals & gait modes
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: Autonomous Locomotion Policy (Neural Network / RL or CPG)          │
│  Rate: 20 – 50 Hz                                                           │
│  • Host / Edge inference: Lightweight MLP (ONNX runtime or PyTorch)         │
│  • Inputs: High-level [vx, vy, yaw], real-time IMU (roll/pitch/yaw), joints │
│  • Outputs: Continuous target joint angles / PD offsets                     │
│  • Responsibilities: Terrain compliance, blind stair traversal, dynamic      │
│    push recovery, foot contact timing, slip prevention                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Joint angle targets (ASCII / binary)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 0: Microcontroller Firmware & Reflexes (OpenCat on NyBoard/BiBoard)   │
│  Rate: 100 – 500 Hz                                                         │
│  • Real-time MPU6050 IMU gyro/accelerometer interrupts                      │
│  • Onboard balance reflex (OpenCat `kbalance` / tilt compensation)          │
│  • PWM servo driver (PCA9685) & hardware safety angle clamps                │
│  • Hardware E-stop (T_REST torque release)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Obstacle Course & Stairs: Physical Constraints vs. Simulation

### Physical Scale Reality of Petoi Bittle
From `models/bittle.urdf` and CAD specifications:
- **Body Dimensions**: Length = 204.4 mm, Width = 109.9 mm, Standing Height = 88.2 mm.
- **Limb Segments**: Upper leg = ~42 mm, Lower leg = ~43 mm.
- **Maximum Step Clearance**: During standard trotting or walking gaits, foot ground clearance is only **15 mm to 25 mm** (~0.6 to 1.0 inch).
- **The "Human Stair" Problem**:
  - A standard residential stair riser is **180 mm to 200 mm** high. That is more than **twice the height of the entire Bittle dog**!
  - Bittle **cannot** climb standard human stairs. Any attempt to step onto an 18cm riser will result in the robot walking directly into a vertical wall.

### What an Obstacle Course for Bittle Actually Looks Like
To provide a realistic and exciting challenge, the course should feature:
1. **Modular Mini-Stairs / Terraces**:
   - Step Riser (height): **15 mm to 22 mm** (graduating up to 25 mm).
   - Step Tread (depth): **70 mm to 100 mm** (allows full four-paw stance).
   - Challenge: Center-of-mass (CoM) pitching upward; rear paws must maintain traction while front paws reach onto the step.
2. **Inclined Ramps**:
   - **15° to 25° slope**.
   - Challenge: Forward pitch compensation during ascent, backward pitch compensation during descent to prevent face-planting.
3. **Low Clearance Tunnel / Underpass**:
   - **65 mm to 75 mm ceiling** (Bittle stands at 88 mm).
   - Challenge: Forces GLM to transition the robot into a belly crawl (`crF`) or crouch posture, crawl through, and stand back up.
4. **Elevated Balance Beam / Bridge**:
   - **80 mm to 100 mm width**.
   - Challenge: Yaw drift correction; any slip of the outer paws triggers edge recovery.
5. **Slalom Cones & Stepping Stones**:
   - Spaced 150mm apart; tests path-following, turning radius, and precise stepping.

---

## 3. How Far Can We Take This? (The Evolutionary Roadmap)

### Milestone 1: 3D WebGL Obstacle Course in `bittle-agent` (Interactive Twin)
- **What it does**:
  - Add modular 3D obstacles (mini-stairs, ramps, tunnels, balance beams, slalom cones) to `static/js/viewer.js` in Three.js.
  - Expose course control to the UI and GLM:
    - `bittle_load_course({preset: "mini_stairs" | "ramp_bridge" | "tunnel_crawl" | "agility_slalom"})`
    - `bittle_get_course_state()` returning obstacle positions and Bittle's position/collisions.
  - GLM can compose or draft custom movesets specifically designed to solve each obstacle, previewing the trajectory in the 3D twin.
- **Effort**: Low to Medium (builds directly on existing Three.js + `app/agent.py`).

### Milestone 2: PyBullet / Rigid-Body Physics Sim & Contact Evaluation
- **What it does**:
  - Integrate a PyBullet environment (`sim_bullet.py`) loaded from `models/bittle.urdf`.
  - Simulate real gravity, friction, joint torque, and collision dynamics.
  - When GLM generates a novel moveset, it executes in PyBullet first:
    - If the robot tips over or slips on a stair, PyBullet reports physics metrics (pitch/roll tilt, foot contact forces, energy expenditure).
    - GLM inspects the failure telemetry, iterates on the keyframe angles/timings, and re-tests until the gait succeeds.
- **Effort**: Medium.

### Milestone 3: Neural Locomotion Policy (RL + CPG)
- **What it does**:
  - Train an end-to-end quadruped locomotion policy in PyBullet / Isaac Gym using PPO (Stable-Baselines3).
  - Train over randomized terrain: stairs, ramps, heightfields, and lateral pushes.
  - Export the trained policy as a lightweight ONNX model running at 50 Hz.
  - GLM controls the policy via velocity and behavior vectors:
    - `cmd_vel(vx=0.15, vy=0.0, yaw=0.2)`
    - `set_gait_parameters(step_height=0.03, body_height=0.07)`
- **Effort**: High.

### Milestone 4: Multi-Modal Vision-Language-Action (VLA)
- **What it does**:
  - Using a camera (webcam pointing at the physical course or simulated camera in PyBullet/Three.js).
  - A local Vision-Language Model (e.g. GLM-4V or Qwen2-VL on DGX Spark) inspects the visual frame:
    - Detects obstacles and estimates relative distance: *"Stair 1 is 15cm ahead, 18mm high."*
    - Directs navigation and triggers specialized obstacle-crossing sequences autonomously.
- **Effort**: High.

---

## 4. Proposed Implementation Architecture for Milestone 1

### File Structure & Worktree
Work will be developed on the dedicated git worktree:
`bittle-agent/.worktrees/wt-obstacle-course` (branch: `obstacle-course`)

### Proposed Changes

#### Component: Frontend 3D Viewer & Simulation (`static/js/`)
- **[NEW] `static/js/obstacle_course.js`**:
  - Modular Three.js obstacle generator:
    - `MiniStairs(stepCount, riserHeight, treadDepth)`
    - `InclineRamp(length, height, angle)`
    - `CrawlTunnel(width, height, length)`
    - `BalanceBridge(length, width, elevation)`
    - `SlalomPoles(count, spacing)`
  - Collision & ground height query helper (`getTerrainHeightAt(x, z)` and bounding box collision checks).
- **[MODIFY] `static/js/viewer.js`**:
  - Add `loadCourse(presetName)`, `clearCourse()`, and ground height clamping so Bittle's feet and torso physically interact with elevated surfaces during playback.
  - Toggle for displaying obstacle courses in the viewport.

#### Component: Agent Harness & Tools (`app/agent.py` & `app/motion/`)
- **[MODIFY] `app/agent.py`**:
  - Update `SYSTEM_PROMPT` with knowledge of obstacle courses, stair clearance limits (max 20-25mm), ramp physics, and crawl tunnel postures.
  - Add new tools:
    - `bittle_load_course`: Select and spawn an obstacle scenario (e.g. "mini_stairs", "ramp_bridge", "crawl_tunnel").
    - `bittle_get_course_layout`: Inspect current obstacle dimensions and coordinates.
    - `bittle_evaluate_terrain_clearance`: Check if a candidate moveset clears the obstacle height profile without foot collision or tipping.
- **[MODIFY] `app/motion/builtin_library.py`**:
  - Add specialized obstacle movesets:
    - `stair_step_up`: Dynamic pitch-up, front-paw ledge plant, CoM weight transfer, rear tuck.
    - `ramp_climb`: Low-stance forward crawl with aggressive knee flexion.
    - `low_tunnel_crawl`: 60mm clearance belly crawl.

---

## 5. Open Questions & User Clarifications

1. **Immediate Focus**: Would you like to start with **Milestone 1** (building the interactive 3D WebGL obstacle course & stair challenge directly into the existing Three.js viewer and GLM agent harness), or would you prefer jumping straight into setting up a **PyBullet / Python rigid-body physics simulation** backend?
2. **Physical Robot Setup**: Do you have a physical Bittle that you plan to test on real-world obstacles (e.g. 3D printed / cardboard stairs), or are we primarily optimizing the simulation and autonomous agent discovery first?
3. **Hardware / Board**: If you have a physical Bittle, is it equipped with the **NyBoard (ATmega328P)** or the **BiBoard (ESP32 / Bittle X)**, and do you have any sensors attached (ultrasonic distance sensor, camera, or standard gyro only)?
