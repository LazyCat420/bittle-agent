# Petoi Bittle 3D Models & Simulation Assets

This directory contains verified 3D models and robot description files for the Petoi Bittle quadruped robot.

## Contents

- **`bittle_assembled.stl`**: Full, assembled CAD model of the Petoi Bittle quadruped (276,266 triangles).
  - **Interactive 3D on GitHub**: GitHub automatically renders this file in an interactive 3D WebGL viewport directly in your web browser. Click the file above to rotate, zoom, and inspect the physical assembly without downloading extra software.
  - **Dimensions**: Length = 204.4 mm, Width = 109.9 mm, Height = 88.2 mm.
- **`bittle.urdf`**: Authoritative URDF (Universal Robot Description Format) specification.
  - Formatted for ROS 1 / ROS 2 (`robot_state_publisher`, RViz, Gazebo), Isaac Sim, PyBullet, and MuJoCo URDF loaders.
  - Contains accurate link inertial tensors, masses, visual mesh links (pointing to `../static/assets/meshes/`), and revolute joint limits verified against `app/joints.py` and OpenCat firmware.

## Kinematic Overview

| Joint Name | OpenCat Index | Parent Link | Child Link | Joint Axis | Limits (rad) | Limits (deg) |
|---|---|---|---|---|---|---|
| `neck_joint` | 0 | `base_link` | `head_link` | `[0, 0, 1]` | `[-2.094, 2.094]` | `[-120°, 120°]` |
| `shrfs_joint` | 9 | `base_link` | `rf_shoulder_link` | `[0, 1, 0]` | `[-2.234, 1.396]` | `[-128°, 80°]` |
| `shrft_joint` | 13 | `rf_shoulder_link` | `rf_knee_link` | `[0, 1, 0]` | `[-1.396, 2.217]` | `[-80°, 127°]` |
| `shlfs_joint` | 8 | `base_link` | `lf_shoulder_link` | `[0, 1, 0]` | `[-2.234, 1.396]` | `[-128°, 80°]` |
| `shlft_joint` | 12 | `lf_shoulder_link` | `lf_knee_link` | `[0, 1, 0]` | `[-1.396, 2.217]` | `[-80°, 127°]` |
| `shrrs_joint` | 10 | `base_link` | `rr_shoulder_link` | `[0, 1, 0]` | `[-1.396, 2.217]` | `[-80°, 127°]` |
| `shrrt_joint` | 14 | `rr_shoulder_link` | `rr_knee_link` | `[0, 1, 0]` | `[-1.396, 2.217]` | `[-80°, 127°]` |
| `shlrs_joint` | 11 | `base_link` | `lr_shoulder_link` | `[0, 1, 0]` | `[-1.396, 2.217]` | `[-80°, 127°]` |
| `shlrt_joint` | 15 | `lr_shoulder_link` | `lr_knee_link` | `[0, 1, 0]` | `[-1.396, 2.217]` | `[-80°, 127°]` |

## Rebuilding Assets

To regenerate the STL and URDF models from the base CAD meshes:
```bash
npm run models
# or
python3 scripts/build_models.py
```
