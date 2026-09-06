#!/usr/bin/env python3
"""
Petoi Bittle 3D Model Builder & Export Utility.

Generates:
1. models/bittle_assembled.stl: Unified binary STL model rendered natively in GitHub's web 3D viewer.
2. models/bittle.urdf: Standard URDF robot description for ROS 1/2, Gazebo, Isaac Sim, and RViz.

Zero external dependencies (pure Python standard library).
"""

import math
import os
import struct
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MESH_DIR = os.path.join(REPO_ROOT, "static", "assets", "meshes")
MODELS_DIR = os.path.join(REPO_ROOT, "models")


def parse_obj(file_path):
    """Parse vertices and triangular faces from an OBJ file."""
    vertices = []
    triangles = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                # Vertex coordinates (in mm in CAD export)
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                # Face indices (1-indexed)
                idx = []
                for p in parts[1:]:
                    v_idx = int(p.split("/")[0])
                    if v_idx < 0:
                        v_idx = len(vertices) + v_idx + 1
                    idx.append(v_idx - 1)
                # Fan triangulate polygons with > 3 vertices
                for i in range(1, len(idx) - 1):
                    triangles.append((idx[0], idx[i], idx[i + 1]))

    return vertices, triangles


def compute_normal(v1, v2, v3):
    """Compute normalized surface normal for triangle (v1, v2, v3)."""
    ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
    bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag > 1e-12:
        return (nx / mag, ny / mag, nz / mag)
    return (0.0, 0.0, 1.0)


def build_assembled_stl(output_path):
    """Merge all 37 Bittle CAD meshes in rest assembly into a single binary STL."""
    mesh_names = [
        "base_link.obj", "battery_1.obj", "cover_1.obj", "front__1.obj", "rear__1.obj",
        "c_neck__1.obj", "servo_neck__1.obj", "head__1.obj", "jaw_1.obj",
        "servo_rfs_1.obj", "c_thrf__1.obj", "th_rf_1.obj", "tho_rf__1.obj", "tube__1.obj", "servos_rf_1.obj", "shank_rf_1.obj",
        "servo_lfs_1.obj", "c_thlf_1.obj", "th_lf_1.obj", "tho_lf_1.obj", "tube_lf_1.obj", "servos_lf_1.obj", "shank_lf_1.obj",
        "servo_rrs__1.obj", "c_thrr_1.obj", "th_rr_1.obj", "tho_rr_1.obj", "tube_rr_1.obj", "servos_rr_1.obj", "shank_rr_1.obj",
        "servo_lrs__1.obj", "c_thlr_1.obj", "th_lr__1.obj", "tho_lr_1.obj", "tube_lr_1.obj", "servos_lr_1.obj", "shank_lr_1.obj"
    ]

    all_triangles = []
    min_b = [float("inf"), float("inf"), float("inf")]
    max_b = [float("-inf"), float("-inf"), float("-inf")]

    for name in mesh_names:
        p = os.path.join(MESH_DIR, name)
        if not os.path.isfile(p):
            print(f"Warning: {name} not found, skipping")
            continue
        verts, tris = parse_obj(p)
        for i1, i2, i3 in tris:
            v1, v2, v3 = verts[i1], verts[i2], verts[i3]
            norm = compute_normal(v1, v2, v3)
            all_triangles.append((norm, v1, v2, v3))
            for v in (v1, v2, v3):
                for d in range(3):
                    min_b[d] = min(min_b[d], v[d])
                    max_b[d] = max(max_b[d], v[d])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        header = b"Petoi Bittle Assembled Model - High Fidelity Digital Twin (LazyCat420)"
        f.write(header.ljust(80, b"\0")[:80])
        f.write(struct.pack("<I", len(all_triangles)))

        for norm, v1, v2, v3 in all_triangles:
            f.write(struct.pack("<3f", *norm))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<3f", *v3))
            f.write(struct.pack("<H", 0))

    print(f"Generated {output_path} ({len(all_triangles)} triangles, {os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"  Bounds (mm): X=[{min_b[0]:.1f}, {max_b[0]:.1f}], Y=[{min_b[1]:.1f}, {max_b[1]:.1f}], Z=[{min_b[2]:.1f}, {max_b[2]:.1f}]")
    print(f"  Dimensions: Length={max_b[0]-min_b[0]:.1f}mm, Width={max_b[1]-min_b[1]:.1f}mm, Height={max_b[2]-min_b[2]:.1f}mm")


def build_urdf(output_path):
    """Generate standard URDF model with exact kinematic coordinates and joint limits."""
    urdf_content = """<?xml version="1.0"?>
<!-- Petoi Bittle Quadruped Robot URDF
     Kinematics verified against Petoi CAD and OpenCat firmware.
     Scale: 1.0 (meters for coordinates; meshes scaled by 0.001 from mm).
-->
<robot name="bittle">

  <material name="bittle_yellow">
    <color rgba="0.96 0.72 0.0 1.0"/>
  </material>
  <material name="bittle_blue">
    <color rgba="0.12 0.25 0.66 1.0"/>
  </material>
  <material name="bittle_red">
    <color rgba="0.84 0.16 0.16 1.0"/>
  </material>
  <material name="bittle_dark">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>

  <!-- Base Link / Torso Chassis -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <mass value="0.108"/>
      <inertia ixx="0.00015" ixy="0" ixz="0" iyy="0.00006" iyz="0" izz="0.00021"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/base_link.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_yellow"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/front__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/rear__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/cover_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/battery_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_neck__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servo_rfs_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servo_lfs_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servo_rrs__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servo_lrs__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
  </link>

  <!-- Head Assembly (Joint 0) -->
  <link name="head_link">
    <inertial>
      <origin xyz="0.02 0 0.02" rpy="0 0 0"/>
      <mass value="0.0187"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
    <visual>
      <origin xyz="-0.047554 0 -0.035941" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/head__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_yellow"/>
    </visual>
    <visual>
      <origin xyz="-0.047554 0 -0.035941" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/jaw_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
    <visual>
      <origin xyz="-0.047554 0 -0.035941" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servo_neck__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
  </link>

  <joint name="neck_joint" type="revolute">
    <origin xyz="0.047554 0 0.035941" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="head_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0944" upper="2.0944" effort="1.0" velocity="5.58"/>
  </joint>

  <!-- ─── Front-Right Leg ─────────────────────────────────────── -->
  <link name="rf_shoulder_link">
    <inertial>
      <origin xyz="-0.02 0 -0.005" rpy="0 0 0"/>
      <mass value="0.020"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
    <visual>
      <origin xyz="-0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_thrf__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/th_rf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tho_rf__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_thorf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tube__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
  </link>

  <joint name="shrfs_joint" type="revolute">
    <origin xyz="0.0525 -0.0485 0.022" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="rf_shoulder_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.234" upper="1.396" effort="1.0" velocity="5.58"/>
  </joint>

  <link name="rf_knee_link">
    <inertial>
      <origin xyz="0.02 0 0" rpy="0 0 0"/>
      <mass value="0.014"/>
      <inertia ixx="0.000005" ixy="0" ixz="0" iyy="0.000005" iyz="0" izz="0.000005"/>
    </inertial>
    <visual>
      <origin xyz="-0.008067 0.04708 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servos_rf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="-0.008067 0.04708 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/shank_rf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
  </link>

  <joint name="shrft_joint" type="revolute">
    <origin xyz="-0.044433 0.00142 -0.011906" rpy="0 0 0"/>
    <parent link="rf_shoulder_link"/>
    <child link="rf_knee_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.396" upper="2.217" effort="1.0" velocity="5.58"/>
  </joint>

  <!-- ─── Front-Left Leg ──────────────────────────────────────── -->
  <link name="lf_shoulder_link">
    <inertial>
      <origin xyz="-0.02 0 -0.005" rpy="0 0 0"/>
      <mass value="0.020"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
    <visual>
      <origin xyz="-0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_thlf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/th_lf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tho_lf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_tholf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="-0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tube_lf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
  </link>

  <joint name="shlfs_joint" type="revolute">
    <origin xyz="0.0525 0.0485 0.022" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="lf_shoulder_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.234" upper="1.396" effort="1.0" velocity="5.58"/>
  </joint>

  <link name="lf_knee_link">
    <inertial>
      <origin xyz="0.02 0 0" rpy="0 0 0"/>
      <mass value="0.014"/>
      <inertia ixx="0.000005" ixy="0" ixz="0" iyy="0.000005" iyz="0" izz="0.000005"/>
    </inertial>
    <visual>
      <origin xyz="-0.008067 -0.04708 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servos_lf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="-0.008067 -0.04708 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/shank_lf_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
  </link>

  <joint name="shlft_joint" type="revolute">
    <origin xyz="-0.044433 -0.00142 -0.011906" rpy="0 0 0"/>
    <parent link="lf_shoulder_link"/>
    <child link="lf_knee_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.396" upper="2.217" effort="1.0" velocity="5.58"/>
  </joint>

  <!-- ─── Rear-Right Leg ──────────────────────────────────────── -->
  <link name="rr_shoulder_link">
    <inertial>
      <origin xyz="-0.02 0 -0.005" rpy="0 0 0"/>
      <mass value="0.020"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
    <visual>
      <origin xyz="0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_thrr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/th_rr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tho_rr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_thorr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tube_rr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
  </link>

  <joint name="shrrs_joint" type="revolute">
    <origin xyz="-0.0525 -0.0485 0.022" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="rr_shoulder_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.396" upper="2.217" effort="1.0" velocity="5.58"/>
  </joint>

  <link name="rr_knee_link">
    <inertial>
      <origin xyz="0.02 0 0" rpy="0 0 0"/>
      <mass value="0.014"/>
      <inertia ixx="0.000005" ixy="0" ixz="0" iyy="0.000005" iyz="0" izz="0.000005"/>
    </inertial>
    <visual>
      <origin xyz="0.096933 0.04708 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servos_rr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0.096933 0.04708 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/shank_rr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
  </link>

  <joint name="shrrt_joint" type="revolute">
    <origin xyz="-0.044433 0.00142 -0.011906" rpy="0 0 0"/>
    <parent link="rr_shoulder_link"/>
    <child link="rr_knee_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.396" upper="2.217" effort="1.0" velocity="5.58"/>
  </joint>

  <!-- ─── Rear-Left Leg ───────────────────────────────────────── -->
  <link name="lr_shoulder_link">
    <inertial>
      <origin xyz="-0.02 0 -0.005" rpy="0 0 0"/>
      <mass value="0.020"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
    <visual>
      <origin xyz="0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_thlr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/th_lr__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tho_lr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/c_tholr__1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
    <visual>
      <origin xyz="0.0525 -0.0485 -0.022" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/tube_lr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_red"/>
    </visual>
  </link>

  <joint name="shlrs_joint" type="revolute">
    <origin xyz="-0.0525 0.0485 0.022" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="lr_shoulder_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.396" upper="2.217" effort="1.0" velocity="5.58"/>
  </joint>

  <link name="lr_knee_link">
    <inertial>
      <origin xyz="0.02 0 0" rpy="0 0 0"/>
      <mass value="0.014"/>
      <inertia ixx="0.000005" ixy="0" ixz="0" iyy="0.000005" iyz="0" izz="0.000005"/>
    </inertial>
    <visual>
      <origin xyz="0.096933 -0.047082 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/servos_lr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_dark"/>
    </visual>
    <visual>
      <origin xyz="0.096933 -0.047082 -0.010094" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../static/assets/meshes/shank_lr_1.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bittle_blue"/>
    </visual>
  </link>

  <joint name="shlrt_joint" type="revolute">
    <origin xyz="-0.044433 -0.001418 -0.011906" rpy="0 0 0"/>
    <parent link="lr_shoulder_link"/>
    <child link="lr_knee_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.396" upper="2.217" effort="1.0" velocity="5.58"/>
  </joint>

</robot>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(urdf_content)
    print(f"Generated {output_path} ({os.path.getsize(output_path)} bytes)")


def main():
    stl_path = os.path.join(MODELS_DIR, "bittle_assembled.stl")
    urdf_path = os.path.join(MODELS_DIR, "bittle.urdf")

    print("=== Building Bittle 3D Models ===")
    build_assembled_stl(stl_path)
    build_urdf(urdf_path)
    print("=== Build Complete ===")


if __name__ == "__main__":
    main()
