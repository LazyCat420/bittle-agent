# Plan & Root-Cause Fix: Dynamic Gravity & Kinematic Ground Contact for Bittle

## 1. Problem Statement
When Bittle is commanded to `sit` (or `rest`, `pu`, `bf`, etc.), the 3D digital twin executes the leg joint movements in mid-air while the torso chassis remains pinned at the standing height ($y = 0.0532\text{m}$). As a result, Bittle floats in the air and kicks its legs up instead of having gravity pull its pelvis/rump down to sit firmly on the ground surface.

---

## 2. Expected vs. Actual Behavior

### Expected Behavior
- When the user or GLM commands **"sit"**:
  1. Front legs articulate to planting stance (knees ~40°, shoulders ~-30°), remaining grounded.
  2. Rear legs fold forward and upward (knees ~75°, shoulders ~80°), retracting the hind paws.
  3. Under gravity, the rear chassis/rump drops down until it sits firmly on the ground surface ($y \approx 0.022\text{m}$).
  4. The torso pitches upward realistically by $\sim 10^\circ$ to $15^\circ$, resting the dog in an authentic seated posture.
- When commanded to **"rest"**:
  - The entire robot sinks to the ground, with its belly resting flat at $y \approx 0.020\text{m}$.
- When commanded to **"balance" / "stand"**:
  - The legs push against the ground, elevating the chassis back to $y = 0.0532\text{m}$.

### Actual Behavior
- In `static/js/viewer.js`, line 960:
  `const targetY = 0.0532 + (yFront + yRear) / 2.0;`
  `this.robotGroup.position.y = THREE.MathUtils.lerp(this.robotGroup.position.y, targetY, 0.18);`
- The base height constant `0.0532` is hardcoded and ignores joint angles.
- When `sit` is executed, the rear paws fold up into the body, but the chassis stays hovering $53.2\text{mm}$ in the air, leaving the rear rump floating $35\text{mm}$ above the floor.

---

## 3. First-Principles Model & Invariants

### Invariants that must hold:
1. **Gravity Non-Penetration Invariant**: The lowest physical contact point of the robot (front paws/chest, and rear paws/rump) must always touch the terrain surface ($y = \text{terrainY}$). No part of the robot may float in the air without support, nor penetrate through the floor.
2. **Kinematic Reach Feedback**: Torso height and pitch must be functions of current joint angles $\vec{\theta}$, not static constants.
3. **Continuity & Settling Invariant**: Transitions between postures (e.g. stand $\to$ sit $\to$ stand) must smoothly settle at a physical rate without sudden popping.

### Kinematic Ground Contact Equations:
For any joint angle state $\vec{\theta}$:
- **Front Contact Reach**:
  $$h_{\text{front}}(\vec{\theta}) = \max\left(d_{\text{chest\_min}}, \text{reach}_{\text{FL}}(\theta_8, \theta_{12}), \text{reach}_{\text{FR}}(\theta_9, \theta_{13})\right)$$
- **Rear Contact Reach**:
  $$h_{\text{rear}}(\vec{\theta}) = \max\left(d_{\text{rump\_min}}, \text{reach}_{\text{RL}}(\theta_{11}, \theta_{15}), \text{reach}_{\text{RR}}(\theta_{10}, \theta_{14})\right)$$

Where:
- $d_{\text{chest\_min}} \approx 0.020\text{m}$ (bottom of front chassis)
- $d_{\text{rump\_min}} \approx 0.022\text{m}$ (bottom of rear pelvis/battery)
- $\text{reach}_{\text{leg}}(\theta_s, \theta_k) = -z_{\text{foot}}(\theta_s, \theta_k)$ computed from leg forward kinematics.

Ground heights:
$$Y_{\text{front\_target}} = y_{\text{terrain\_front}} + h_{\text{front}}(\vec{\theta})$$
$$Y_{\text{rear\_target}} = y_{\text{terrain\_rear}} + h_{\text{rear}}(\vec{\theta})$$
$$Y_{\text{chassis\_target}} = \frac{Y_{\text{front\_target}} + Y_{\text{rear\_target}}}{2}$$
$$\theta_{\text{pitch\_target}} = \arctan\left(\frac{Y_{\text{front\_target}} - Y_{\text{rear\_target}}}{L_{\text{wheelbase}}}\right)$$

### Posture Contact Table:
| Posture | Front Reach $h_f$ | Rear Reach $h_r$ | Chassis Height $Y_c$ | Torso Pitch $\theta_{\text{pitch}}$ | Physical Appearance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stand / Balance** | $0.0532\text{m}$ | $0.0532\text{m}$ | $0.0532\text{m}$ | $0.0^\circ$ | Upright on all 4 paws |
| **Sit** | $0.0480\text{m}$ | $0.0220\text{m}$ | $0.0350\text{m}$ | $+12.4^\circ$ (nose up) | Front paws down, butt on floor |
| **Rest (Belly)** | $0.0200\text{m}$ | $0.0200\text{m}$ | $0.0200\text{m}$ | $0.0^\circ$ | Flat rested on floor |
| **Push-Up Down** | $0.0250\text{m}$ | $0.0532\text{m}$ | $0.0391\text{m}$ | $-13.1^\circ$ (nose down) | Chest near floor, rear standing |
| **Stretch** | $0.0250\text{m}$ | $0.0600\text{m}$ | $0.0425\text{m}$ | $-16.2^\circ$ (bow) | Play bow / morning stretch |

---

## 4. Root-Cause Analysis (Symptom vs. Cause)

- **Symptom**: When commanding `sit`, the legs move into sitting angles, but Bittle hovers in mid-air.
- **Root Cause**: `viewer.js` calculates elevation using a hardcoded standing height `0.0532` in the render loop. It lacks a dynamic leg-reach forward kinematics model and lacks pelvis/rump contact boundaries.

---

## 5. Ranked Hypotheses

1. **Hypothesis 1 (Confirmed - Kinematic Contact Elevation Solver)**:
   Replacing the static `0.0532` constant with dynamic forward kinematics $h_{\text{front}}(\vec{\theta})$ and $h_{\text{rear}}(\vec{\theta})$ and a chassis ground contact floor ($d_{\text{min}} = 0.020\text{m}$) directly satisfies the contact equations for all postures (sit, stand, rest, pushups, stretch) in real time.
2. **Hypothesis 2 (Rigid Body Physics Engine in WebGL - Ammo.js/Cannon.js)**:
   Embedding a full rigid body physics engine in Three.js with capsule colliders for all 15 mesh links.
   *Why rejected*: Heavy (adds 2MB WASM dependency), prone to physics instability/jitter on servo joints, whereas kinematic contact grounding is deterministic, jitter-free, and lightweight.

---

## 6. Proposed Changes

### Component: 3D Twin Viewer (`static/js/viewer.js`)

#### [`viewer.js`](file:///home/lazycat/github/projects/sun/bittle-agent/static/js/viewer.js)
1. **Add Leg Forward Kinematics Reach Evaluator**:
   ```javascript
   computeLegReach(shoulderAngleDeg, kneeAngleDeg, isRear = false)
   ```
   Computes vertical drop from shoulder to foot using the exact link lengths ($L_{\text{thigh}} = 46.02\text{mm}$, $L_{\text{shank}} = 48.0\text{mm}$).
2. **Add Dynamic Ground Contact Solver**:
   ```javascript
   computeGroundContactOffsets(angles)
   ```
   Evaluates front reach $h_f$ and rear reach $h_r$ with chassis contact minimums ($20\text{mm}$ for belly/rump).
3. **Update Render Loop in `animate()`**:
   Replace static `targetY = 0.0532 + terrainY` with:
   ```javascript
   const { hFront, hRear } = this.computeGroundContactOffsets(this.currentAngles);
   const yFrontTarget = yFront + hFront;
   const yRearTarget = yRear + hRear;
   const targetY = (yFrontTarget + yRearTarget) / 2.0;
   const targetPitch = Math.atan2(yFrontTarget - yRearTarget, 0.120);
   ```
4. **Update `setEstop` and `resetPose`**:
   Remove manual height hacks; let the contact solver naturally drop the robot when angles change.

### Component: Python Kinematics & Ground Contact Model (`app/motion/`)

#### [`obstacles.py`](file:///home/lazycat/github/projects/sun/bittle-agent/app/motion/obstacles.py)
- Add `compute_posture_ground_contact(angles)` to mirror the contact model in Python so that GLM and simulation tools have identical ground contact telemetry.

---

## 7. Verification & Regression Plan

### Automated Tests
1. **Ground Contact Height Test**:
   - `pytest tests/test_ground_contact.py`:
     - Test `STAND_ANGLES` produces height $\approx 53.2\text{mm}$, pitch $\approx 0^\circ$.
     - Test `SIT_ANGLES` produces rear contact at rump ($\approx 22\text{mm}$), front paws at $\approx 48\text{mm}$, pitch $> 10^\circ$.
     - Test `REST_ANGLES` produces height $\approx 20\text{mm}$, pitch $\approx 0^\circ$.
     - Test `pu` (pushups) produces front dip ($\approx 25\text{mm}$), rear $\approx 53\text{mm}$, negative pitch.

### Manual Verification
1. Open the Bittle Web UI.
2. Click or command **"Sit"**: Observe Bittle's rear body smoothly sink under gravity until its butt sits flat on the ground, with front paws firmly planted and chest angled up.
3. Command **"Balance"** or **"Stand"**: Observe Bittle push up with its legs back to full standing height.
4. Command **"Rest"**: Observe Bittle drop completely flat onto its belly on the ground.
5. On **Mini Stairs**: Command Bittle to sit on the top landing platform; observe both rump and paws ground cleanly.
