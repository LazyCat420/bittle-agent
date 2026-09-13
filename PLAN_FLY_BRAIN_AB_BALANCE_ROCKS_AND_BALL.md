# Rigorous A/B/C/D Experimental Plan: Fly Brain vs. Conventional Reflex Locomotion & Dynamic Balancing for Bittle

## 1. Executive Summary & Problem Formulation
This revised experimental plan governs the benchmark within [bittle-agent](file:///home/lazycat/github/projects/sun/bittle-agent) to rigorously evaluate whether biological circuitry derived from the fruit fly (*Drosophila melanogaster*) connectome provides measurable computational advantage over classical control methods in quadruped stabilization.

The initial plan suffered from a critical methodological flaw: **it conflated the benefit of adding an auxiliary stabilizer with the benefit of the fly connectome itself.** Furthermore, it assumed the fly Ellipsoid Body / Protocerebral Bridge ($EB/PB$) ring attractor naturally tracks a 2D sphere surface normal (which contradicts the neurobiological literature on 1-DoF angular integration) and relied on unphysical "tilt counter-torques" that bypass Petoi Bittle's position-controlled servo actuation.

This revised protocol establishes:
1. **Four Controlled Arms** (PPO alone, PPO + Conventional Reflex, PPO + Fly Circuit, PPO + Rewired Null Circuit).
2. **Physically Grounded Actuation** enforcing Petoi Bittle's 8-servo position limits, rate limits, and torque constraints.
3. **Rigorous Ball & Terrain Physics** using thin spherical shell inertia ($I = \frac{2}{3}mr^2$), full 6-DoF rolling/torsional contact dynamics, and validated static rock terrain.
4. **Strict Observability Separation** ensuring deployable onboard IMU sensors are distinguished from privileged simulation state.
5. **Staged Evidence Progression** (Physics Gate $\to$ 3M-step Pilot $\to$ 5-seed $\times$ 30M-step Multi-Run Evaluation with bootstrap 95% confidence intervals).

---

## 2. Claim Classification & Verification Matrix (VCPM)

| ID | Claim Statement | Classification | Evidence / Source / Validation Gate |
|---|---|---|---|
| **C-01** | Bittle is actuated by 8 position-controlled servos with bounded range, slew limits ($6^\circ/\text{step}$ at $50\text{ Hz}$), and torque ceiling ($0.25\text{ N}\cdot\text{m}$). Arbitrary body torques cannot be commanded. | **Verified Fact** | Verified in `trainer/env/spec.py#L29-L34`, `trainer/assets/build_models.py`, and [Petoi Python API](https://docs.petoi.com/apis/python-api). |
| **C-02** | `terrain.level = 2` in `trainer/env/terrain.py` generates fixed yaw-only half-buried boxes ($10\text{–}30\text{ mm}$ height), not independently moving rocks or rolling stones. | **Verified Fact** | Verified by direct inspection of `trainer/env/terrain.py#L40-L50` and `trainer/config.py#L200-L215`. |
| **C-03** | A hollow basketball shell ($m = 0.6\text{ kg}, r = 0.12\text{ m}$) has moment of inertia $I = \frac{2}{3}mr^2 = 0.00576\text{ kg}\cdot\text{m}^2$, 40% higher than a solid sphere ($\frac{2}{5}mr^2 = 0.003456\text{ kg}\cdot\text{m}^2$). | **Verified Fact** | Classical rigid-body mechanics; verified by formula derivation. |
| **C-04** | Drosophila $EB/PB$ ring attractors encode a 1-DoF circular heading angle $\theta \in [0, 2\pi)$ for angular integration, which cannot natively represent a 2-DoF surface normal on $S^2$. | **Verified Fact** | [Green et al., Nature 2017](https://www.nature.com/articles/nature22343); Turner-Evans et al., eLife 2017. |
| **C-05** | Adding a conventional PD attitude feedback reflex (Arm B) improves rock-course completion over baseline PPO alone (Arm A). | **Testable Claim** | Compare 100 evaluation rollouts on fixed `terrain.level = 2` seeds. |
| **C-06** | An engineered fly-derived recurrent circuit (Arm C) improves course completion or balance survival over the conventional reflex (Arm B) by $\ge 15\%$. | **Testable Claim** | 5-seed comparative evaluation with non-overlapping 95% bootstrap CIs ($p < 0.01$). |
| **C-07** | A degree- and weight-matched randomized rewired null circuit (Arm D) fails to match Arm C's performance. | **Testable Claim** | Permutation test across paired evaluation seeds ($p < 0.01$). |

---

## 3. Four Experimental Comparison Arms

Every arm receives identical observation vectors, action clipping envelopes, and $50\text{ Hz}$ update frequencies.

```
                  ┌──────────────────────────────────────────────┐
                  │ Deployable Observation (41-dim)              │
                  │ • IMU Gravity vector g (3)                   │
                  │ • Scaled Gyroscope ω (3)                     │
                  │ • Velocity Command (3)                       │
                  │ • Target History (8 joints x 3 steps = 24)   │
                  │ • Last Action (8)                            │
                  └──────────────────────┬───────────────────────┘
                                         │
       ┌──────────────────┬──────────────┴─────┬──────────────────┐
       ▼                  ▼                    ▼                  ▼
┌──────────────┐   ┌──────────────┐     ┌──────────────┐   ┌──────────────┐
│    ARM A     │   │    ARM B     │     │    ARM C     │   │    ARM D     │
│  PPO Alone   │   │ PPO + PD     │     │ PPO + Fly    │   │ PPO + Rewired│
│  (Baseline)  │   │ Conventional │     │ Circuit      │   │ Null Connect.│
└──────┬───────┘   └──────┬───────┘     └──────┬───────┘   └──────┬───────┘
       │                  │                    │                  │
       │                  ├─ + Δθ_PD           ├─ + Δθ_Fly        ├─ + Δθ_Null
       ▼                  ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Action Pipeline: Δθ_total = Clip(Δθ_PPO + Δθ_reflex, -6°, +6°)          │
│ Applied to Position-Controlled Servos with Low-Pass Damping & Saturation│
└─────────────────────────────────────────────────────────────────────────┘
```

### Arm Definitions
1. **Arm A (Baseline PPO Alone):**
   * Standard Brax PPO actor-critic network (2 hidden layers, 256 units, tanh activation).
   * Learns gait and balance end-to-end without auxiliary stabilization controllers.
2. **Arm B (PPO + Conventional Feedback Reflex):**
   * PPO policy identical in architecture to Arm A.
   * **Conventional Reflex Layer:** A classical Proportional-Derivative (PD) attitude leveling controller.
     $$\Delta \theta_{\text{reflex, PD}} = -K_p \cdot \vec{e}_{\text{tilt}} - K_d \cdot \vec{\omega}_{\text{gyro}}$$
     where $\vec{e}_{\text{tilt}} = \vec{g}_{\text{nominal}} - \vec{g}_{\text{measured}}$.
   * Corrective trim is mapped into bounded joint-angle offsets $\Delta \theta \in [-6^\circ, +6^\circ]$ (crouching stance on tilt, differential shoulder trim to counter roll/pitch).
3. **Arm C (PPO + Engineered Fly-Derived Reflex):**
   * PPO policy identical in architecture to Arm A.
   * **Engineered Fly Circuit:**
     * **Input:** Gyro angular velocity $\vec{\omega}$ and gravity projection $\vec{g}$.
     * **Internal State:** Coupled dual-ring attractor network (Roll Attractor $\Phi_{\text{roll}}$, Pitch Attractor $\Phi_{\text{pitch}}$) driven by angular acceleration $\frac{d\vec{\omega}}{dt}$ (Haltere sensory model) and angular velocity integration.
     * **Decoder:** Linear projection from ring-bump position and phase velocity to joint trim $\Delta \theta_{\text{reflex, Fly}} \in [-6^\circ, +6^\circ]$.
4. **Arm D (PPO + Randomized Null Network Control):**
   * Exactly matches Arm C in neuron count, synapse count, sparsity, and activation functions, but with synaptic connections randomized via degree-preserving Maslov-Sneppen edge swapping.
   * Isolates whether performance gains in Arm C arise from specific biological connectome topologies or merely from having an additional recurrent reservoir.

### Training & Co-Adaptation Protocol
* **Primary Regime (Co-Trained):** The reflex controller (Arm B, C, or D) is active **during training**. PPO rollouts experience the reflex dynamics from step 0, allowing the RL policy to learn complementary gait strategies.
* **Secondary Ablation (Zero-Shot Post-Hoc Transfer):** Take the trained Arm A policy (frozen weights) and evaluate it with Arm B, C, and D reflex layers applied post-hoc to measure drop-in stabilization without retraining.
* **Non-Interference Rule:** Co-trained and post-hoc evaluation results are reported in strictly separate tables.

---

## 4. Actuation, Observability, and Physical Realism

### 4.1 Bittle Joint Actuation Model
* Bittle has **8 active joints**: Front-Left (Shoulder, Knee), Front-Right (Shoulder, Knee), Rear-Left (Shoulder, Knee), Rear-Right (Shoulder, Knee).
* Actuation is strictly **position-controlled**; no body wrench or direct torques are allowed.
* Per-step target delta is bounded by the hardware envelope:
  $$\Delta \theta_j \in [-6^\circ, +6^\circ] \quad (\pm 0.1047\text{ rad/step} = \pm 5.24\text{ rad/s at } 50\text{ Hz})$$
* Servo dynamics model:
  $$\tau_j = \text{clip}\left(k_p (\theta_j^{\text{cmd}} - \theta_j) - d \dot{\theta}_j, \ -0.25\text{ N}\cdot\text{m}, \ +0.25\text{ N}\cdot\text{m}\right)$$
  with $k_p = 10.0\text{ N}\cdot\text{m/rad}$ and $d = 0.05\text{ N}\cdot\text{m}\cdot\text{s/rad}$.

### 4.2 Sensor Observability Specification
* **Deployable Onboard Observation (Actor Input - 41 dims):**
  * Projected gravity vector $\vec{g} / ||\vec{g}||$ (3 dims)
  * Gyroscope angular velocity $\vec{\omega} \times 0.25$ (3 dims)
  * Target velocity command $\vec{v}_{\text{cmd}} / \text{scale}$ (3 dims)
  * Target angle history $(\theta_{\text{hist}} - \theta_{\text{stand}}) / 60^\circ$ (24 dims: 8 joints $\times$ 3 steps)
  * Last executed action (8 dims)
* **Privileged State (Critic Input Only - Asymmetric PPO):**
  * For Rocks: 3x3 local height scan around torso, base linear velocity in world frame, foot contact forces.
  * For Basketball: Sphere 3D position $(x, y, z)$, sphere linear velocity $(\dot{x}, \dot{y}, \dot{z})$, sphere angular velocity $(\omega_{bx}, \omega_{by}, \omega_{bz})$, contact patch normal vector.
  * **Strict Isolation:** Privileged state is NEVER exposed to the actor or reflex layers.

---

## 5. Environment Specifications & Physics Validation

### 5.1 Suite A: Walking on Rocks (`rough_walk`)
* **Terrain Generation:** Uses `bittle-agent`'s verified `terrain.kind = "rough"`, `terrain.level = 2`.
* **Geometry:** Static half-buried boxes protruding $15\text{–}30\text{ mm}$ above the floor, with random yaw $\pm 45^\circ$, box half-sizes $20\text{–}60\text{ mm}$, and random spacing $120\text{ mm}$.
* **Runway:** $5.0\text{ m}$ length, $0.4\text{ m}$ width. Spawn jitter $\pm 0.05\text{ m}$.
* **Physics Parity:** Training in MuJoCo Warp GPU; benchmark evaluation in CPU MuJoCo mesh model (`bittle_cpu_terrain.xml`).

### 5.2 Suite B: The Basketball Balance Challenge (`ball_balance`)
* **Ball Asset (`trainer/assets/generated/bittle_basketball.xml`):**
  * **Geometry:** Sphere radius $r = 0.12\text{ m}$ (standard size 7 basketball radius).
  * **Mass:** $m = 0.60\text{ kg}$.
  * **Inertia Matrix:** Thin hollow spherical shell:
    $$I_{xx} = I_{yy} = I_{zz} = \frac{2}{3} m r^2 = \frac{2}{3} (0.60) (0.12)^2 = 0.00576\text{ kg}\cdot\text{m}^2$$
  * **Contacts & Friction (MuJoCo `condim=6`):**
    * Sliding friction: $\mu_s = 1.0$ (rubber paw on basketball pebble surface).
    * Torsional friction: $\mu_t = 0.005$.
    * Rolling friction: $\mu_r = 0.005$.
    * Contact compliance: `solref = [0.015, 1.0]`, `solimp = [0.9, 0.95, 0.001]`.
  * **Floor:** Flat rubber mat ($z = -0.12\text{ m}$) with friction $\mu = 1.0$.
* **Spawn Stance:** Bittle initialized standing symmetrically on top of the sphere with 4 paws pre-planted at equilibrium contact angles ($\pm 25^\circ$ spread from vertical).

---

## 6. Success Metrics & Diagnostic Measurements

| Task | Primary Outcome Metric | Supporting & Diagnostic Measurements | Termination / Failure Thresholds |
|---|---|---|---|
| **Rocks (`rough_walk`)** | **Course Completion Rate (%)**: Percentage of 100 evaluation runs reaching the 5.0m line without falling. | • Forward progress distance (m)<br>• Falls per meter ($m^{-1}$)<br>• Paw slip velocity (m/s)<br>• Actuator saturation rate (% of steps at limit) | • Torso/head contact $> 0.2\text{ s}$<br>• $|roll| > 60^\circ$ or $|pitch| > 60^\circ$<br>• Max episode time: $20\text{ s}$ ($1000\text{ steps}$) |
| **Basketball (`ball_balance`)** | **Time Balanced (s)**: Duration Bittle maintains balance up to fixed horizon $T_{\max} = 20.0\text{ s}$ ($1000\text{ steps}$). | • Torso-to-ball radial offset $||\vec{r}_{\text{torso}} - \vec{r}_{\text{ball}}||_{xy}$ (m)<br>• Ball drift from origin $||\vec{r}_{\text{ball}}||_{xy}$ (m)<br>• Settling time after $0.5\text{ N}$ lateral push impulse<br>• Actuator saturation rate (%) | • Torso contact with ball or floor<br>• Any foot losing contact with sphere $> 0.3\text{ s}$<br>• $|roll| > 50^\circ$ or $|pitch| > 50^\circ$ |

---

## 7. Staged Evidence & Execution Roadmap

```
Stage 1: Physics & Kinematic Validation (0 training steps)
  ├── 1.1 Assert thin-shell basketball inertia & contact friction in MuJoCo
  ├── 1.2 Assert servo rate limits and joint angle envelope clipping
  └── 1.3 Assert terrain.level = 2 rock generation determinism
Stage 2: Short Pilot Training Runs (3M transition steps, 1 seed per arm)
  ├── 2.1 Smoke test Arm A, B, C, D learning curves
  ├── 2.2 Verify absence of policy freeze or NaN gradient instability
  └── 2.3 Check sample efficiency trajectories
Stage 3: Full Multi-Seed Comparative Campaign (5 seeds x 30M steps per arm)
  ├── 3.1 Execute 5 independent seeds for Arms A, B, C, D on Rocks
  ├── 3.2 Execute 5 independent seeds for Arms A, B, C, D on Basketball
  └── 3.3 Run 100 held-out evaluation rollouts per seed; compute bootstrap 95% CIs
```

### Clarification on Step Counting
* "30M steps" means **30,000,000 environment transition steps total per seed** collected across 1,024 parallel vectorized environments ($~29,296$ steps per environment).
* 4 arms $\times$ 5 seeds $\times$ 30M steps = 600M environment transitions per suite.
* Running Stage 2 (3M steps) first guarantees we do not burn GPU compute if an arm has a structural bug.

### Pre-Defined Minimum Effect Size Gate
To claim that the biological fly connectome provides superior stabilization:
1. **Arm C must outperform Arm B (Conventional Reflex)** on the primary outcome metric by **$\ge 15\%$**.
2. The difference must be statistically significant ($p < 0.01$, paired bootstrap test with non-overlapping 95% confidence intervals).
3. **Arm C must outperform Arm D (Rewired Null)** by $\ge 10\%$ ($p < 0.01$) to prove topological causality.
If Arm C matches Arm B or ties Arm D, the biological hypothesis is falsified.

---

## 8. Open Questions & Alignment Check

1. **Reflex Regime:** Does the proposed focus on **co-training** (with post-hoc transfer reported as a secondary ablation) match your expectations?
2. **Compute Allocation:** Does running the 3M-step Stage 2 pilots across Arms A–D before committing to the full 5-seed $\times$ 30M-step Stage 3 campaign work for your RTX 3090 Ti availability?
3. **Worktree Directory:** Shall we proceed with setting up the isolated worktree at `bittle-agent/.worktrees/wt-fly-balance` on branch `feat/fly-reflex-balance` when you give the word?
