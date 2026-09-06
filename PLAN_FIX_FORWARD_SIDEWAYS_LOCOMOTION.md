# Plan: Fix Forward & Backward Sideways Locomotion (Crab-Walking)

## 1. Problem Statement & Intake
- **Bug**: When commanding Bittle forward or backward (via D-Pad `▲ FWD`, `W` key, or GLM agent tool `bittle_navigate_step`), the robot walks sideways (crab-walks to its right on forward, and to its left on backward) instead of forward and backward.
- **Expected Behavior**: Bittle's body must translate along the line of its nose (forward) when commanded forward, and translate along the line of its tail (backward) when commanded backward.
- **Actual Behavior**: The visual mesh of Bittle faces World $-Z$, but the locomotion displacement equations update `robotPosition.x += vx * dt * Math.cos(this.robotYaw)` along World $+X$. Because World $+X$ is orthogonal ($90^\circ$) to Bittle's nose, Bittle moves sideways.
- **Steps to Reproduce**:
  1. Open viewer UI.
  2. Press `W` or hold `▲ FWD`.
  3. Observe that Bittle walks towards its right flank (+X) while its nose faces -Z.

---

## 2. Invalidation of Prior Attempt (Rule 0 & Rule 2)
- **Prior Test**: `test_viewer_kinematic_equations` in `tests/test_locomotion_displacement.py`.
  - **Classification**: **False Positive Type 1 (Decoupled)**.
  - **Reason**: Tested the displacement math in Python without asserting the 3D mesh heading vector in Three.js. It passed while the visual twin walked sideways.
- **Prior Assumption Disproven**: *"Setting `torso.rotation.x = -Math.PI / 2; torso.rotation.z = Math.PI / 2;` in `viewer.js` points Bittle toward $+X$."*
  - **Disproof**: Direct vector transformation of CAD nose vector $(1, 0, 0)$ under Euler rotation `(-pi/2, 0, pi/2)` yields $(0, 0, -1)$ — pointing to World $-Z$. Dot product with forward displacement $+X$ is $0.000$ ($90^\circ$ orthogonal).

---

## 3. System Model & Broken Invariants
1. **Heading Invariant**: $\vec{v}_{\text{nose}} \cdot \vec{v}_{\text{forward}} > 0$ and $\vec{v}_{\text{nose}} \cdot \vec{v}_{\text{lateral}} = 0$.
   - Broken in `viewer.js:728`: `torso.rotation.z = Math.PI / 2` pointed the nose along $-Z$ while forward displacement moves along $+X$.
2. **Terrain & Pitch Invariant**: Terrain elevation probes and pitch tilt must align with the robot's heading.
   - Broken in `viewer.js:1125-1144`: Terrain probes sampled front at $z - 0.053$ (assuming nose at $-Z$), and pitch rotation was applied to `rotation.x` (acting as roll instead of pitch when heading is $+X$).
3. **Obstacle Course Invariant**: The obstacle courses (`mini_stairs`, `ramp_bridge`, `agility_slalom`), backend tool `bittle_navigate_step`, and step riser colliders (`startX = 0.140`) are all aligned along $+X$.

---

## 4. Architectural Fix Options

### Option 1 (Recommended): Align 3D Visual Mesh to +X Forward Axis
- In `viewer.js`, set `torso.rotation.z = 0` (removing the $90^\circ$ offset so CAD $+X$ maps directly to World $+X$).
- Update pitch conformation to Euler order `'YZX'` and rotate around local $Z$ (the lateral shoulder axis when facing $+X$).
- Update terrain elevation front/rear sampling probes along $+X$:
  `frontX = x + 0.053 * Math.cos(yaw)`, `frontZ = z - 0.053 * Math.sin(yaw)`.
- **Pros**: Perfectly unifies visual mesh with obstacle courses (`mini_stairs` at $+X$), backend navigation tools (`bittle_navigate_step` advancing $+X$), step riser colliders, and camera presets.
- **Cons**: Requires minor update to `tests/test_ground_plane_collision.py` to match `torso.rotation.z = 0`.

### Option 2: Re-architect the Entire World & Backend to -Z Forward Axis
- Keep visual mesh facing $-Z$, and change locomotion math to move along $-Z$.
- Re-architect `obstacle_course.js`, `app/motion/obstacles.py`, `app/agent.py`, step collision risers, and camera presets to generate everything along $-Z$ instead of $+X$.
- **Pros**: Keeps `torso.rotation` in `viewer.js` unchanged.
- **Cons**: Massive cross-stack blast radius touching backend, tools, obstacles, and existing tests. High regression risk.

---

## 5. Verification Plan

### Automated Tests
1. **New Three.js Mesh Alignment Test (`tests/test_mesh_locomotion_alignment.py`)**:
   Asserts that Three.js node hierarchy head vector has $\vec{v}_{\text{nose}} \cdot (1, 0, 0) > 0.70$ and lateral component $= 0.0$.
2. **Ground Plane Zero-Penetration Test (`tests/test_ground_plane_collision.py`)**:
   Verifies with `torso.rotation.z = 0` that absolute minimum Y is $\ge 0.0$ across all poses and walking frames.
3. **Locomotion Displacement Suite (`tests/test_locomotion_displacement.py`)**:
   Verifies forward, backward, left, and right stepping.
4. **Full Test Suite**:
   Run all 148+ pytest unit and integration tests.

### Manual Verification
1. Run `npm run deploy` on Synology NAS.
2. Open `http://nas:8008`.
3. Press `W` / `▲ FWD`: Verify Bittle moves straight forward towards the stairs.
4. Press `S` / `▼ REV`: Verify Bittle retreats straight backward away from the stairs.
5. Press `A` / `D` (Left / Right): Verify Bittle turns cleanly while stepping.
