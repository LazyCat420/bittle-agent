# Root-Cause Diagnostic & Fix Plan: 3D Terrain Rendering in Simulation Viewport

**Worktree:** `/home/lazycat/github/projects/sun/bittle-agent/.worktrees/wt-fix-terrain`  
**Branch:** `fix/terrain-render-framing`  
**Status:** Investigation Complete & Verified via Browser Diagnostics — Awaiting User Approval (Order of Operations Step 1)

---

## 1. Problem Statement (Three-Part Investigation per `/fix` Rule 1)

* **Expected Behavior:** When the user selects a terrain preset from the **Terrain** dropdown (e.g. *Mini Stairs*, *Ramp & Bridge*, *Crawl Tunnel*, *Agility Slalom*), the 3D obstacle course should immediately render in full view directly in front of the Bittle robot, framed clearly within the Three.js viewport, with high visual contrast against the dark background.
* **Actual Behavior:** When selecting any terrain preset, the 3D canvas appears completely unchanged. The robot remains alone on the flat grid, giving the visual appearance that terrain switching failed to render.
* **Steps to Reproduce:**
  1. Open `http://10.0.0.16:8008/ui/` in a browser.
  2. Locate the **Terrain** dropdown in the 3D viewport control bar (`#courseSelect`).
  3. Change the selection from *"Flat Ground"* to *"Mini Stairs (18mm Steps)"*.
  4. Observe the 3D viewport — no stairs or obstacles appear on screen.

---

## 2. Invalidation of Prior Attempt & False Positive Log (`/fix` Rule 0 & 2)

### Prior Passing Tests Declared Void:
* **`tests/test_obstacles.py` (127 passing tests):** Verified that Python mathematical dictionaries (`COURSE_LAYOUTS`), clearance formulas, and API endpoints returned HTTP 200 with valid schema.
* **Why this was a False Positive (Type 1 & Type 2 Decoupled):** The Python unit tests tested schema data in isolation. They did not test Three.js world coordinates, camera frustum clipping, or WebGL rendering. Passing backend tests masked the fact that 3D meshes were rendered completely outside the camera's field of view.
* **Status:** Flagged as *Do Not Reuse as proof of visual rendering*.

### Disproven Assumptions:
| Wrong Assumption | Evidence Disproving It |
| :--- | :--- |
| **"JavaScript or Three.js crashed when loading the terrain"** | Browser subagent inspection confirmed **zero errors, zero warnings, and zero exceptions** in the browser console. The `ObstacleCourse.loadPreset()` function executed completely and meshes were added to the scene. |
| **"Obstacles were placed in front of Bittle along the +Z axis"** | Vector analysis of `viewer.js:466` proved Bittle's forward axis (head) points along **+X** (world coordinates), while obstacles were placed along **+Z** (`x = 0, z = 0.16m..0.54m`). The obstacles were rendered at Bittle's right flank, 90° off-axis. |
| **"The default camera at `(0.35, 0.28, 0.38)` shows the obstacle course"** | Three.js perspective camera calculation (FOV 45°, lookAt `(0, 0.06, 0)`) confirmed that points at `z >= 0.35m` have projected coordinates `screen_x = -0.68` to `-1.66` and `screen_y = -0.45` to `-0.96`, which are **completely outside the view frustum**. |
| **"Obstacle materials (0x1c2128) are clearly visible"** | Screenshots showed that with `FogExp2(0x0e1116, 1.2)` and a dark floor `0x090c10`, even when manually zoomed out, the obstacles had near-zero contrast. |

---

## 3. First-Principles System Model & Invariants

```
[User Selects Course] 
       │
       ▼
[onCourseSelected(name)] ──► [viewer.loadCourse(name)] 
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
[Mesh Generation]                                      [Camera Viewport]
• Obstacle geometry generated along Forward Axis      • Frustum must encompass
• High-contrast PBR materials & safety markings         Robot + Obstacle bounds
• Placed directly ahead of robot's nose               • Smooth framing transition
```

### Core Invariants That Must Hold:
1. **Heading Invariant:** The obstacle course forward axis MUST match Bittle's forward heading axis. Bittle's snout must face directly into the entrance of the course at `start_dist`.
2. **Frustum Invariant:** Whenever an obstacle course is active, the camera viewport frustum MUST encompass both the robot and the full length of the obstacle course without manual user panning or zooming.
3. **Contrast Invariant:** Obstacle geometry MUST possess high luminance contrast (>4.5:1 relative luminance) and emissive boundary cues against the dark `0x0e1116` environment and `FogExp2`.

---

## 4. Root Causes Identified

1. **Heading Axis Disconnect (+X vs +Z):**
   * Bittle's CAD torso was rotated: `torso.rotation.x = -Math.PI / 2; torso.rotation.z = Math.PI / 2;`.
   * In Three.js world space, this places Bittle's head along **+X**, tail along **-X**, right legs along **+Z**, and left legs along **-Z**.
   * However, `obstacle_course.js` and `COURSE_LAYOUTS` generated obstacles along **+Z** (`stepMesh.position.set(0, centerY, centerZ)`).
   * Result: Obstacles sprouted out of Bittle's right ribs at a 90° angle rather than in front of its face.

2. **Camera Frustum Clipping & Absence of Dynamic Framing:**
   * Default camera: `camera.position.set(0.35, 0.28, 0.38)`, `controls.target.set(0, 0.06, 0)`, `fov = 45°`.
   * This camera position is a tight close-up on the 200mm robot at the origin.
   * Obstacles spanning 160mm to 780mm along the floor are positioned outside the 45° frustum cone and clipped by the viewport boundaries.
   * When the user selects a terrain, the camera target was never adjusted to frame the course.

3. **Inverted Camera Presets in `viewer.js`:**
   * In `viewer.js:514-532`:
     * `front` preset was set to `(0, 0.12, 0.45)` (looking down +Z) — but looking down +Z looks directly into Bittle's *side ribs*, not its front!
     * `side` preset was set to `(0.45, 0.12, 0)` (looking down +X) — which looks directly into Bittle's *nose*, not its side!

4. **Low Visual Luminance & Fog Absorption:**
   * Obstacle materials used dark charcoal hex colors (`0x1c2128`, `0x0f141c`, `0x21262d`).
   * Scene fog (`FogExp2(0x0e1116, 1.2)`) rapidly attenuates objects beyond 0.3m.
   * Without high-visibility hazard stripes or contrasting step materials, the obstacles blended into the floor.

---

## 5. Architectural Options for Resolution (Rule 3: Clarify Multiple Ways)

### Option A: Standardize on +X as Forward (Recommended)
* **How it works:**
  * Keep Bittle's existing CAD hierarchy facing **+X** (forward).
  * Update `obstacle_course.js` so obstacles are placed along the **+X axis** in front of Bittle: `startX = 0.160m`, stepping forward towards `+X`.
  * Update `COURSE_LAYOUTS` in `app/motion/obstacles.py` to use `start_x_m`, `min_x`, `max_x`, matching the physics coordinate system.
  * Correct camera presets: `front` looks from `(+0.45, 0.12, 0)`, `side` looks from `(0, 0.12, 0.45)`.
  * Add automatic camera framing `viewer.fitCameraToCourse(presetName)`: when a course is chosen, the camera smoothly animates or sets its target to the center of the course/robot combo `(targetX ~ 0.30m, targetY ~ 0.06m, targetZ ~ 0m)` with distance expanded to `~0.85m` to provide a wide, dramatic view.
  * When `none` (Flat Ground) is selected, reset camera target to `(0, 0.06, 0)` at distance `0.58m`.
* **Pros:**
  * Retains Bittle's natural orientation in the CAD assembly without touching complex servo rotation pivot math.
  * Perfectly fixes camera presets (`front` actually faces front, `side` faces side).
  * Direct alignment between 3D visualization and kinematic forward walking along +X.

### Option B: Rotate Bittle to Face +Z
* **How it works:**
  * Modify `torso.rotation` in `viewer.js` (`rx = -Math.PI/2, ry = -Math.PI/2, rz = 0`) so Bittle's head points along **+Z**.
  * Keep `obstacle_course.js` placing obstacles along **+Z**.
  * Keep `COURSE_LAYOUTS` using `z` coordinates.
  * Add automatic camera framing along the Z axis.
* **Cons of Option B:**
  * Touching `torso.rotation` alters the CAD axis mapping of all 8 leg joints (shoulder pitch and knee pitch rotation axes in `jointNodes`), requiring re-verifying all joint rotations and animations to ensure legs don't invert or rotate on the wrong axis.
  * High regression risk for existing gait animations.

**Recommendation:** **Option A** is far cleaner, more robust, and isolates the fix to obstacle coordinate generation and camera framing without disturbing joint kinematics.

---

## 6. Proposed Changes Breakdown (Option A)

### Component: Frontend 3D Viewer & Obstacle Course
#### `static/js/obstacle_course.js`
* Reorient all 4 obstacle builders (`buildMiniStairs`, `buildRampBridge`, `buildCrawlTunnel`, `buildAgilitySlalom`) to generate along the **+X axis**:
  * Step boxes: `BoxGeometry(treadDepth, currentElevation, stepWidth)` placed at `(centerX, centerY, 0)`.
  * Warning stripes along each step edge.
  * Ramp wedges and bridge planks aligned along X.
  * Crawl tunnel arch and semi-transparent roof spanning along X with glowing entry arch.
  * Agility slalom cones arranged along X with alternating `±Z` lateral offsets.
* Upgrade material palette with premium high-visibility styling:
  * Sleek industrial slate/titanium tread finish (`0x2d3748` / `0x4a5568`) with bright neon hazard striping (`0xffa500` / `0x00f2fe`).
  * Self-illuminating waypoint markers and entry gates.
* Update `getElevationAt(x, z)` and `checkCollision(x, y, z)` to query the X-axis bounds.

#### `static/js/viewer.js`
* Implement `fitCameraToCourse(presetName)`:
  * Computes the bounding box of the active obstacle course + robot.
  * Adjusts `controls.target` and `camera.position` to frame both robot and obstacle course with optimal padding.
  * Provides immediate, dramatic visibility the moment the user selects any option in `#courseSelect`.
* Correct `setCameraPreset(name)`:
  * `'front'`: `camera.position.set(0.48, 0.12, 0)` looking at `(0, 0.06, 0)`.
  * `'side'`: `camera.position.set(0, 0.12, 0.48)` looking at `(0, 0.06, 0)`.
  * `'iso'`: `camera.position.set(0.42, 0.32, 0.45)`.
* In `loadCourse(presetName)`:
  * Call `obstacleCourse.loadPreset(presetName)`.
  * Trigger `fitCameraToCourse(presetName)`.

#### `static/index.html`
* Add visual status feedback in the activity log and viewport overlay when terrain is loaded.

### Component: Backend Motion & Obstacle Logic
#### `app/motion/obstacles.py`
* Update `COURSE_LAYOUTS` coordinates to use `start_x_m`, `min_x`, `max_x`, `y` elevations, and `±z` lateral cone offsets so backend collision verification perfectly mirrors the 3D viewport.
* Update `evaluate_clearance()` for X-forward progression.

#### `tests/test_obstacles.py`
* Update automated tests for X-axis coordinate bounds, ensuring all 127+ unit tests pass.

---

## 7. Verification Plan

### Automated Tests:
1. Run pytest suite:
   ```bash
   cd /home/lazycat/github/projects/sun/bittle-agent/.worktrees/wt-fix-terrain
   /home/lazycat/github/projects/sun/bittle-agent/.venv/bin/python -m pytest tests/test_obstacles.py -v
   /home/lazycat/github/projects/sun/bittle-agent/.venv/bin/python -m pytest tests -q
   ```
2. Verify coordinate bounds and clearance algorithms match the new +X orientation.

### Manual Verification (User-Facing):
1. Push branch to GitHub:
   ```bash
   git push origin fix/terrain-render-framing
   ```
2. Deploy container to Synology NAS:
   ```bash
   npm run deploy
   ```
3. Test in browser (`http://10.0.0.16:8008/ui/`):
   * Select **Mini Stairs**: Verify 3 stairs appear directly in front of Bittle's snout, fully framed, with bright hazard stripes.
   * Select **Ramp & Bridge**: Verify ramp incline leads up directly from Bittle's front feet to the elevated beam.
   * Select **Crawl Tunnel**: Verify tunnel aligns with Bittle's forward path.
   * Select **Agility Slalom**: Verify cones zig-zag along Bittle's forward path.
   * Select **Flat Ground**: Verify camera returns to clean close-up inspection view.
