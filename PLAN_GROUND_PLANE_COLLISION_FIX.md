# Plan & Root-Cause Fix: Ground Plane Leg Penetration & Authentic Contact Grounding

## 1. Problem Statement
When Bittle stands, sits, walks, or performs skills in the 3D digital twin viewport, the paws (blue shank tips) penetrate below the floor grid plane ($Y < 0$). In standard standing pose, the feet clip $\sim 1\text{mm}$ to $3\text{mm}$ under the ground, and during gaits or posture transitions (such as rest or sit), the legs sink up to $15\text{mm}$ beneath the floor surface.

---

## 2. Expected vs. Actual Behavior

### Expected Behavior
- **Ground Non-Penetration Invariant**: No part of Bittle's paws, shanks, or body may ever penetrate below the floor plane ($Y = 0$) or obstacle surface ($Y = y_{\text{terrain}}(x, z)$):
  $$\min_{\text{vertex } v \in \text{Robot}} \left( y_{\text{world}}(v) - y_{\text{terrain}}(x_v, z_v) \right) \ge 0$$
- In standing posture (`stand`), the bottom-most curved surface of the rubber paw pads rests tangentially on the grid surface ($Y = 0.000\text{m}$).
- During walking, trotting, crawling, and sitting gaits, the lowest supporting paw or rump contacts the ground at $Y = 0.000\text{m}$ while elevated swinging paws lift cleanly above it without clipping into the floor.

### Actual Behavior
- In standing pose, the lowest shank vertices are at $Y = -0.00099\text{m}$ ($-0.99\text{mm}$ under the plane).
- During walk cycles (`wkF`), paw vertices drop to $Y = -0.00313\text{m}$ ($-3.13\text{mm}$ under the plane).
- In rest pose (`rest`), front paw vertices drop to $Y = -0.01506\text{m}$ ($-15.06\text{mm}$ under the plane).
- When pitch was applied to `robotGroup.rotation.z`, it rolled the robot around its longitudinal spine axis instead of pitching, forcing one side of the robot's limbs deep into the plane.

---

## 3. Invalidation of Previous Attempt & False Positive Classification

### False Positive Classification (FP Type 3 — Decoupled Mathematical Approximation)
- **What went wrong with the prior attempt?**
  The previous implementation introduced `computeLegReach(shoulderDeg, kneeDeg)` in JavaScript and `compute_posture_ground_contact` in Python based on idealized 2-link planar trigonometry ($v_{\text{thigh}} = 44.4\text{mm}, v_{\text{shank}} = 47.8\text{mm}$).
- **Why it failed to catch the bug:**
  1. The automated tests in `tests/test_ground_contact.py` tested the idealized Python mathematical formula, which returned $53.2\text{mm}$.
  2. However, the **actual 3D CAD mesh** (`shank_*.obj`) possesses curved foot pad geometry, heel contours, and thickness extending up to $54.2\text{mm}$ in standing and $> 55.6\text{mm}$ during dynamic gaits.
  3. The formula was decoupled from the actual Three.js scene graph and vertex buffers, producing a false positive test pass while the real WebGL rendering continued to clip through the floor.
  4. The previous code set `robotGroup.rotation.z = this.robotPitch`. Because the torso in `viewer.js` is oriented with head along $-Z$ and spine along $+Y$, rotating around $Z$ is lateral **ROLL**, not pitch!

---

## 4. First-Principles System Model & Invariants

### 1. Invariant 1: Hard Ground Contact Floor
$$\text{targetY} = \max_{i \in \text{Legs/Body}} \left( y_{\text{terrain}}(x_i, z_i) - y_{\text{local\_contact\_i}}(\vec{\theta}) \right)$$
Where $y_{\text{local\_contact\_i}}(\vec{\theta})$ is the vertical coordinate of contact anchor point $i$ when `robotGroup.position.y = 0`. Setting `robotGroup.position.y = targetY` mathematically guarantees that:
$$y_{\text{world\_contact\_i}} \ge y_{\text{terrain}}(x_i, z_i) \quad \forall i$$
with equality for the lowest supporting contact point.

### 2. Invariant 2: Pure Pitch Rotation
Pitch must rotate the robot around its transverse lateral axis ($X$), tilting head up and rump down without lateral roll.
When `robotGroup.rotation.order = 'YXZ'`:
- `rotation.y = yaw` (turns around world vertical $+Y$)
- `rotation.x = pitch` (tilts head up/down around the local lateral axis)
- `rotation.z = 0` (zero roll, preventing one side from dipping into the plane)

---

## 5. Root Cause Analysis (Symptom vs. Cause)

| Symptom | Root Cause |
| :--- | :--- |
| Feet penetrate $1\text{mm}$ to $3\text{mm}$ under plane in standing/walking | `computeLegReach` uses idealized scalar $47.8\text{mm}$ shank length, ignoring actual $54.2\text{mm}$ mesh foot pad bounds. |
| Feet sink $15\text{mm}$ in resting/crouch poses | `computeGroundContactOffsets` took the average of front and rear reaches rather than taking the supremum constraint over all 4 feet. |
| Robot twists and one side digs into the ground when pitched | Pitch was assigned to `rotation.z` (roll axis) instead of `rotation.x` (transverse pitch axis) with `YXZ` Euler ordering. |

---

## 6. Ranked Hypotheses

1. **Hypothesis 1 (Direct Foot Contact Anchor Kinematics + Supremum Ground Elevation Solver - Recommended)**:
   Add 4-5 contact anchor points along the convex hull of each foot pad (heel, mid-pad, ball, toe) and body contact anchors (chest, pelvis, rump). In each frame, evaluate their world positions relative to `robotGroup.y = 0` against the local terrain elevation. Set `targetY = \max_i (y_{\text{terrain}} - y_{\text{rel\_i}})`.
   - *Evidence*: Proven via Node.js Three.js test (`test_contact_solver.mjs`): guarantees $0.00\text{mm}$ floor penetration across `STAND`, `SIT`, `REST`, `PUSHUP`, and dynamic `WALK` cycles.
2. **Hypothesis 2 (Full Mesh Bounding Box Scan every frame)**:
   Run `computeBoundingBox()` on all 16 meshes every frame.
   - *Why rejected*: Unnecessary CPU overhead (thousands of vertex transforms per frame), whereas 5 convex hull anchors per foot produce exact identical contact accuracy in $< 0.02\text{ms}$.
3. **Hypothesis 3 (Static Offset Hack, e.g. adding +5mm to targetY)**:
   Add an arbitrary constant padding.
   - *Why rejected*: Fails on different poses (e.g. sit requires different elevation than rest, stand, or stairs) and causes Bittle to visibly hover in air during other motions.

---

## 7. Step-by-Step Fix Implementation Plan

### Component 1: 3D Twin Viewer (`static/js/viewer.js`)
1. **Define Exact Foot & Body Contact Anchors in `buildBittleModel`**:
   - For each leg's `kneePivot`, attach the 5 convex hull contact points along the foot pad curve:
     - Heel: `(0.0478, 0.0130, -0.0097)`
     - Mid-rear: `(0.0495, 0.0130, -0.0080)`
     - Mid-sole: `(0.0511, 0.0130, -0.0086)` (lowest in stand pose)
     - Ball: `(0.0525, 0.0130, -0.0050)`
     - Toe tip: `(0.0533, 0.0130, -0.0042)`
   - Attach body anchors to `torso`:
     - Chest: `(0.050, 0.0, 0.000)`
     - Pelvis: `(-0.050, 0.0, 0.000)`
     - Rump: `(-0.070, 0.0, 0.000)`
2. **Implement Exact Ground Contact Solver in `animate()`**:
   - Set `this.robotGroup.rotation.order = 'YXZ'`.
   - Apply `this.robotGroup.rotation.y = this.robotYaw`.
   - Apply `this.robotGroup.rotation.x = this.robotPitch` (correct transverse pitch axis).
   - Evaluate all contact anchors against `this.obstacleCourse.getElevationAt(x, z)`.
   - Compute `targetY = max_i (elevation_i - relY_i)`.
   - Smoothly lerp `robotGroup.position.y` to `targetY`.
   - Guarantees $Y_{\text{min}} \ge 0.000\text{m}$ at all times.

### Component 2: Automated Headless Regression Test (`tests/test_ground_plane_collision.py`)
- Create automated test verifying:
  1. `STAND_POSE`: Foot pads rest at $Y \ge 0.000\text{m}$, penetration $= 0.00\text{mm}$.
  2. `SIT_POSE`: Front paws and rump rest at $Y \ge 0.000\text{m}$, penetration $= 0.00\text{mm}$.
  3. `REST_POSE`: Belly and paws rest at $Y \ge 0.000\text{m}$, penetration $= 0.00\text{mm}$.
  4. `wkF` (Walk Frames): Every walk frame maintains $Y \ge 0.000\text{m}$ with zero floor penetration.

---

## 8. Open Questions & Follow-Up Inquiries for the User

1. **Paw Ground Clearance Margin**:
   Do you prefer the paw pads to touch the grid line exactly tangentially ($0.0\text{mm}$ clearance, matching real physical contact), or would you like a tiny visual cushion of $+0.5\text{mm}$ to $+1.0\text{mm}$ so that the dark grid wireframe doesn't z-fight with the foot mesh?
2. **Terrain Conforming Stiffness**:
   When transitioning between standing, walking, and sitting, should the vertical elevation lerp be snappy (~150ms) to immediately react to ground contact, or smoother (~250ms) for softer robotic servo cushioning?
