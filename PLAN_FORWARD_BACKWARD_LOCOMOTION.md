# Plan: Full Forward & Backward Locomotion (Resolving Walking in Place)

## 1. Problem Statement & Intake
When Bittle is commanded to walk or back up (either via skills, movesets, or GLM agent commands), the legs articulate through walking/stepping animations, but the robot's physical position in the 3D scene remains fixed ($X=0, Z=0$) — it is "just walking in place." The user needs Bittle to genuinely move forwards and backwards through the 3D scene across the floor and obstacles.

---

## 2. Root Cause Analysis (Why Bittle Walks in Place)

Through codebase inspection, four distinct root causes were uncovered:

### Root Cause 1: Exact String Mismatch in `getLocomotionSpeed`
In [`viewer.js`](file:///home/lazycat/github/projects/sun/bittle-agent/static/js/viewer.js#L246):
```javascript
  getLocomotionSpeed(sequenceName) {
    const name = (sequenceName || '').toLowerCase();
    if (name.includes('wkf') || name.includes('walk_forward') || name === 'walk') return { vx: 0.080, vyaw: 0 };
    if (name.includes('bk') || name.includes('backup') || name.includes('back')) return { vx: -0.050, vyaw: 0 };
    ...
    return { vx: 0, vyaw: 0 };
  }
```
When skills are run from the Skills & Gaits tab, [`index.html`](file:///home/lazycat/github/projects/sun/bittle-agent/static/index.html#L1749) executes:
```javascript
viewer.playSequence(s.name, { name: s.label || s.name, loop });
```
- In `app/skills.py`, `s.label` is `"Walk forward"`.
- In `viewer.js`, `name = "walk forward"`.
- `"walk forward".includes('walk_forward')` is **FALSE** (space vs underscore).
- `"walk forward".includes('wkf')` is **FALSE**.
- `"walk forward" === 'walk'` is **FALSE**.
- Result: `getLocomotionSpeed("Walk forward")` returns `{ vx: 0, vyaw: 0 }`. The root motion velocity is **zero**, so the legs move while the body remains stationary.

### Root Cause 2: Non-Looping Moveset Preview
In [`index.html`](file:///home/lazycat/github/projects/sun/bittle-agent/static/index.html#L2008) and [`agent-ui.js`](file:///home/lazycat/github/projects/sun/bittle-agent/static/js/agent-ui.js#L339):
Movesets are played with `loop: false`. The standard `wkF` gait has only 4 frames totaling 800ms. Without continuous looping, playback halts almost immediately before meaningful displacement occurs.

### Root Cause 3: Lack of Interactive Directional Controls (Drive Pad & WASD)
Currently, there is no interactive directional pad (D-pad) or keyboard handler (WASD / Arrow keys) for driving Bittle forwards and backwards interactively in the 3D viewport.

### Root Cause 4: Coordinate Alignment with +X Forward Axis
In `viewer.js`, `torso.rotation.z = Math.PI / 2` pointed the head toward World $-Z$ while the locomotion system, obstacle course (stairs/ramps), and navigation tools move along $+X$. When facing $+X$ (`torso.rotation.x = -Math.PI / 2`), $+X$ aligns directly with Bittle's nose.

---

## 3. Proposed Solution Architecture

### 1. Robust Locomotion Velocity Resolver
Expand `getLocomotionSpeed(name)` to normalize spaces, underscores, and token variations:
- **Forward gaits** (`vx = +0.075 m/s`):
  Matches `'wkf'`, `'walk'`, `'forward'`, `'walk forward'`, `'trf'`, `'trot'`, `'stair_step_up'`, `'crf'`, `'crawl'`.
- **Backward gaits** (`vx = -0.055 m/s`):
  Matches `'bk'`, `'back'`, `'backup'`, `'backward'`, `'reverse'`.
- **Left turns** (`vx = +0.025 m/s, vyaw = +0.40 rad/s`):
  Matches `'wkl'`, `'left'`, `'turn_left'`.
- **Right turns** (`vx = +0.025 m/s, vyaw = -0.40 rad/s`):
  Matches `'wkr'`, `'right'`, `'turn_right'`.

### 2. Interactive Manual Drive Controller (UI D-Pad + WASD/Arrow Keys)
Add a dedicated floating Drive Deck on the 3D viewport and keyboard listeners:
- **`W` / `ArrowUp` / `▲ FWD` button**:
  Starts `wkF` in loop mode; advances Bittle forward continuously across the floor.
- **`S` / `ArrowDown` / `▼ REV` button**:
  Starts `bk` in loop mode; steps Bittle backward continuously.
- **`A` / `ArrowLeft` / `◀ LEFT` button**:
  Turns Bittle left continuously while stepping.
- **`D` / `ArrowRight` / `▶ RIGHT` button**:
  Turns Bittle right continuously while stepping.
- **Key release / Stop button**:
  Smoothly halts locomotion and settles Bittle into balanced standing posture (`stand`).

### 3. GLM Agent Discrete Stepping & Distance Integration
Update `bittle_navigate_step` in `app/agent.py` and `agent-ui.js`:
- When GLM commands `"take 3 steps forward"`:
  Play `count` iterations of `wkF`, smoothly displacing Bittle forward by $3 \times 0.038\text{m} = 0.114\text{m}$.
- When GLM commands `"take 2 steps backward"`:
  Play `count` iterations of `bk`, smoothly displacing Bittle backward by $2 \times 0.035\text{m} = 0.070\text{m}$.

### 4. Smooth Camera Following
Keep `cameraFollowEnabled = true` so the Three.js orbit camera smoothly tracks the robot as it moves forward and backward across the scene without losing sight of it.

---

## 4. Verification & Testing Plan

### Automated Tests
1. **Locomotion Velocity Resolver Unit Test (`tests/test_locomotion_speeds.py`)**:
   Test that all skill names and labels (`"Walk forward"`, `"wkF"`, `"Back up"`, `"bk"`, `"Trot"`, `"trF"`, `"Crawl"`) resolve to non-zero velocities in the intended directions:
   - Forward gaits: $v_x > 0$
   - Backward gaits: $v_x < 0$
   - Turn gaits: $v_{\text{yaw}} \ne 0$
2. **End-to-End Three.js Headless Displacement Test (`tests/test_locomotion_displacement.py`)**:
   Run headless simulation of 1.0s of walking:
   - Initial position: $x = 0.000\text{m}$.
   - After 1.0s forward: $x \ge 0.050\text{m}$ (positive forward displacement).
   - After 1.0s backward: $x$ decreases (negative reverse displacement).
   - Verify feet maintain $Y \ge 0$ under the Supremum Ground Solver throughout motion.

### Manual Verification
1. Open `http://nas:8008`.
2. Press and hold `W` or click `▲ Forward`: Bittle moves forward across the floor toward the stairs.
3. Press and hold `S` or click `▼ Backward`: Bittle reverses cleanly away from the stairs.
4. Release key: Bittle halts and stands in balance.
5. In GLM Agent chat, type: *"Take 3 steps forward then 2 steps backward."* Observe the agent step forward, pause, and step backward in the 3D viewport.

---

## 5. Open Questions for User Approval

1. **Keyboard Controls Scope**:
   Should `W/A/S/D` and Arrow key driving be active automatically whenever the 3D viewport has focus, or would you prefer an explicit toggle button (e.g. `🎮 Drive Mode: ON/OFF`) so typing in the chat input doesn't accidentally move the robot? *(Recommended: Active only when chat input is NOT focused)*
2. **Stepping Speed**:
   Do you prefer the forward walk speed set to a realistic moderate pace (~$0.075\text{m/s}$, matching physical Bittle specs) or a faster brisk pace (~$0.12\text{m/s}$)?
