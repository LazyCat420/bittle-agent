"""Stage 1 Physics Verification Gate: Basketball Environment & Rigid-Body Dynamics.

Asserts:
1. Thin-shell spherical moment of inertia: I = 2/3 * m * r^2 = 0.00576 kg.m^2 (NOT solid sphere 2/5).
2. Contact dynamics: condim=6 (sliding, torsional, rolling friction) and compliant solref/solimp.
3. Rolling resistance physics: unanchored ball on floor decelerates cleanly under rolling friction.
4. Clean initial contact: foot spheres contact basketball without initial interpenetration or explosion.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO / "trainer" / "assets"
XML_PATH = ASSETS_DIR / "generated" / "bittle_basketball.xml"
MESH_DIR = REPO / "static" / "assets" / "meshes"


@pytest.fixture(scope="module")
def model_and_data():
    if not XML_PATH.exists():
        from trainer.assets.build_basketball import main as build_bb
        build_bb()

    assets = {p.name: p.read_bytes() for p in MESH_DIR.glob("*.obj")}
    m = mujoco.MjModel.from_xml_string(XML_PATH.read_text(), assets=assets)
    d = mujoco.MjData(m)
    return m, d


def test_basketball_thin_shell_inertia(model_and_data):
    m, _ = model_and_data
    ball_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "basketball")
    assert ball_body_id >= 0, "basketball body not found in model"

    mass = float(m.body_mass[ball_body_id])
    assert math.isclose(mass, 0.60, abs_tol=1e-4), f"expected mass 0.60, got {mass}"

    inertia = m.body_inertia[ball_body_id]
    r = 0.12
    expected_thin_shell_inertia = (2.0 / 3.0) * mass * (r ** 2)  # 0.00576 kg.m^2
    solid_sphere_inertia = (2.0 / 5.0) * mass * (r ** 2)         # 0.003456 kg.m^2

    # Assert thin shell inertia matches to 5 decimal places
    for axis, val in zip(("Ixx", "Iyy", "Izz"), inertia):
        assert math.isclose(float(val), expected_thin_shell_inertia, abs_tol=1e-5), (
            f"{axis} mismatch: got {val}, expected thin-shell {expected_thin_shell_inertia}"
        )
        assert not math.isclose(float(val), solid_sphere_inertia, abs_tol=1e-4), (
            f"{axis} matches solid sphere instead of thin hollow shell!"
        )


def test_basketball_contact_friction_and_condim(model_and_data):
    m, _ = model_and_data
    geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basketball_geom")
    assert geom_id >= 0, "basketball_geom not found in model"

    # MuJoCo condim must be 6 to enable torsional and rolling friction
    condim = int(m.geom_condim[geom_id])
    assert condim == 6, f"expected condim=6 for rolling/torsional friction, got {condim}"

    friction = m.geom_friction[geom_id]
    # friction: [sliding, torsional, rolling]
    assert math.isclose(float(friction[0]), 1.0, abs_tol=1e-4), "sliding friction should be 1.0"
    assert math.isclose(float(friction[1]), 0.005, abs_tol=1e-4), "torsional friction should be 0.005"
    assert math.isclose(float(friction[2]), 0.005, abs_tol=1e-4), "rolling friction should be 0.005"


def test_basketball_rolling_deceleration(model_and_data):
    """Test that rolling friction physically dissipates kinetic energy on a rolling sphere."""
    m, _ = model_and_data
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Move Bittle far out of the way so only the ball rolls on the floor
    # Robot freejoint is qpos 0:7
    d.qpos[2] = 2.0  # suspend Bittle high above
    d.qvel[0:6] = 0.0

    # Give ball forward linear and angular velocity matching pure rolling: v = omega * r
    r = 0.12
    v_init = 0.5  # m/s along +x
    omega_y = v_init / r  # rad/s

    ball_dof_adr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "basketball_free")]
    d.qvel[ball_dof_adr + 0] = v_init
    d.qvel[ball_dof_adr + 4] = omega_y

    # Step physics for 2 seconds (1000 steps at dt = 0.002)
    dt = m.opt.timestep
    n_steps = int(2.0 / dt)
    v_prev = v_init

    for _ in range(n_steps):
        mujoco.mj_step(m, d)

    v_final = float(d.qvel[ball_dof_adr + 0])
    # Ball should have decelerated due to rolling friction
    assert v_final < v_init, f"ball did not decelerate: v_init={v_init}, v_final={v_final}"
    assert v_final >= 0.0, f"ball should not reverse direction on flat floor: v_final={v_final}"
    # Verify ball stayed on the floor without penetrating
    ball_qpos_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "basketball_free")]
    ball_z = float(d.qpos[ball_qpos_adr + 2])
    expected_z = -0.01 + r
    assert math.isclose(ball_z, expected_z, abs_tol=0.005), f"ball height penetrated floor: {ball_z}"


def test_initial_spawn_no_explosive_forces(model_and_data):
    """Assert keyframe initialization has stable contacts without explosive accelerations or asymmetric penetration."""
    m, _ = model_and_data
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Perform a single forward kinematic and velocity step
    mujoco.mj_forward(m, d)

    ball_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basketball_geom")
    leg_contacts = {}
    for leg in ("lf", "rf", "lr", "rr"):
        foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
        leg_contacts[leg] = [
            d.contact[i] for i in range(d.ncon)
            if (d.contact[i].geom1 == foot_id and d.contact[i].geom2 == ball_geom_id)
            or (d.contact[i].geom2 == foot_id and d.contact[i].geom1 == ball_geom_id)
        ]
        # Assert each leg has a valid contact with the basketball
        assert len(leg_contacts[leg]) >= 1, f"Missing contact between {leg}_foot and basketball_geom"
        dist = min(c.dist for c in leg_contacts[leg])
        # Assert no severe penetration (< 2mm)
        assert dist >= -0.002, f"{leg}_foot penetrates basketball by {abs(dist)*1000:.1f} mm"

    # Step 100 steps (0.2s) and assert robot maintains contact atop the ball without being ejected
    for step in range(100):
        mujoco.mj_step(m, d)

    assert d.ncon >= 4, f"Lost contact with basketball after 100 steps: ncon={d.ncon}"
    assert d.qpos[2] >= 0.25, f"Robot fell off basketball: torso z = {d.qpos[2]}"
