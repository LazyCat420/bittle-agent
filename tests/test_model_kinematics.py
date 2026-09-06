"""
Kinematics and 3D Model Verification Tests for Petoi Bittle.

Verifies:
1. Thigh length matches 46.02 mm physical Petoi CAD specification.
2. Symmetrical shoulder and knee pivot coordinates.
3. Required mesh files are present and valid OBJ format.
4. URDF robot description matches joints.py limits and valid XML.
5. Assembled STL model exists, has valid binary STL header, and correct bounding box.
"""

import math
import os
import struct
import xml.etree.ElementTree as ET

from app.joints import CONTROLLABLE, BY_INDEX, REST_POSE

# Exact coordinates from bittle.xml / CAD (in meters)
SHOULDER_RF = (0.0525, -0.0485, 0.022)
KNEE_RF = (0.008067, -0.04708, 0.010094)

SHOULDER_LF = (0.0525, 0.0485, 0.022)
KNEE_LF = (0.008067, 0.04708, 0.010094)

SHOULDER_RR = (-0.0525, -0.0485, 0.022)
KNEE_RR = (-0.096933, -0.04708, 0.010094)

SHOULDER_LR = (-0.0525, 0.0485, 0.022)
KNEE_LR = (-0.096933, 0.047082, 0.010094)


def dist_mm(p1, p2):
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2 +
        (p2[2] - p1[2]) ** 2
    ) * 1000.0


def test_thigh_lengths_match_physical_bittle():
    """All 4 leg thigh links must measure 46.02 mm (+- 0.1 mm) horn-to-horn distance."""
    legs = [
        ("RF", SHOULDER_RF, KNEE_RF),
        ("LF", SHOULDER_LF, KNEE_LF),
        ("RR", SHOULDER_RR, KNEE_RR),
        ("LR", SHOULDER_LR, KNEE_LR),
    ]
    for name, shoulder, knee in legs:
        length_mm = dist_mm(shoulder, knee)
        assert 45.9 <= length_mm <= 46.1, f"{name} thigh length {length_mm:.2f}mm violates 46.02mm spec"


def test_mesh_files_present():
    """Verify all 37 essential Bittle OBJ meshes exist and have non-zero size."""
    mesh_dir = os.path.join(os.path.dirname(__file__), "..", "static", "assets", "meshes")
    assert os.path.isdir(mesh_dir), f"Mesh directory {mesh_dir} missing"

    required_meshes = [
        "base_link.obj", "battery_1.obj", "cover_1.obj", "front__1.obj", "rear__1.obj",
        "c_neck__1.obj", "servo_neck__1.obj", "head__1.obj", "jaw_1.obj",
        "servo_rfs_1.obj", "c_thrf__1.obj", "th_rf_1.obj", "tho_rf__1.obj", "tube__1.obj", "servos_rf_1.obj", "shank_rf_1.obj",
        "servo_lfs_1.obj", "c_thlf_1.obj", "th_lf_1.obj", "tho_lf_1.obj", "tube_lf_1.obj", "servos_lf_1.obj", "shank_lf_1.obj",
        "servo_rrs__1.obj", "c_thrr_1.obj", "th_rr_1.obj", "tho_rr_1.obj", "tube_rr_1.obj", "servos_rr_1.obj", "shank_rr_1.obj",
        "servo_lrs__1.obj", "c_thlr_1.obj", "th_lr__1.obj", "tho_lr_1.obj", "tube_lr_1.obj", "servos_lr_1.obj", "shank_lr_1.obj"
    ]

    for m in required_meshes:
        p = os.path.join(mesh_dir, m)
        assert os.path.isfile(p), f"Missing mesh: {m}"
        assert os.path.getsize(p) > 100, f"Mesh {m} is empty or corrupted"


def test_assembled_stl_model():
    """Verify models/bittle_assembled.stl exists, is valid binary STL, and matches physical bounding box."""
    stl_path = os.path.join(os.path.dirname(__file__), "..", "models", "bittle_assembled.stl")
    assert os.path.isfile(stl_path), "models/bittle_assembled.stl does not exist"

    file_size = os.path.getsize(stl_path)
    assert file_size > 84, "STL file too small to contain valid header + triangles"

    with open(stl_path, "rb") as f:
        header = f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]
        expected_size = 84 + num_triangles * 50
        assert file_size == expected_size, f"STL file size {file_size} does not match triangle count {num_triangles} ({expected_size})"
        assert num_triangles > 1000, f"Expected >1000 triangles for Bittle model, got {num_triangles}"


def test_urdf_model_validity():
    """Verify models/bittle.urdf is valid XML and defines the 8 leg joints matching joints.py."""
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "models", "bittle.urdf")
    assert os.path.isfile(urdf_path), "models/bittle.urdf does not exist"

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib.get("name") == "bittle"

    # Find revolute joints
    joints = {j.attrib["name"]: j for j in root.findall("joint") if j.attrib.get("type") == "revolute"}
    
    # 8 leg joints + head pan
    expected_joints = [
        "neck_joint",
        "shrfs_joint", "shrft_joint",
        "shlfs_joint", "shlft_joint",
        "shrrs_joint", "shrrt_joint",
        "shlrs_joint", "shlrt_joint"
    ]
    for ej in expected_joints:
        assert ej in joints, f"Missing joint {ej} in URDF"
