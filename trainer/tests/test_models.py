import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from trainer.assets import build_models, joint_map as jm  # noqa: E402

GEN = Path(build_models.OUT_DIR)


VARIANTS = ("cpu", "gpu", "cpu_terrain", "gpu_terrain")


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_compile_mass_and_contacts(variant):
    m = mujoco.MjModel.from_xml_path(str(GEN / f"bittle_{variant}.xml"))
    d = mujoco.MjData(m)
    assert m.nu == 8 and m.nq == 15  # neck welded
    assert abs(m.body_subtreemass[m.body("torso").id] - 0.2735) < 0.0005
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert 0.044 < d.qpos[2] < 0.050
    found = [d.sensordata[m.sensor_adr[m.sensor(f"{leg}_foot_floor_found").id]] for leg in jm.LEGS]
    assert all(f > 0 for f in found)
    for leg in jm.LEGS:
        assert abs(d.site(f"{leg}_foot_site").xpos[2]) < 0.003


def test_gpu_variant_has_no_mesh_collisions():
    m = mujoco.MjModel.from_xml_path(str(GEN / "bittle_gpu.xml"))
    mesh_coll = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[i] != 0]
    assert mesh_coll == []


def test_regeneration_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(build_models, "OUT_DIR", tmp_path)
    build_models.main()
    for v in VARIANTS:
        name = f"bittle_{v}.xml"
        assert (tmp_path / name).read_text() == (GEN / name).read_text(), f"{name} drifted: rerun build_models"


def test_joint_map_stand_matches_keyframe():
    m = mujoco.MjModel.from_xml_path(str(GEN / "bittle_cpu.xml"))
    ctrl = jm.stand_ctrl()
    assert np.allclose(ctrl, m.key_ctrl[0], atol=0.1)
    qpos_idx = jm.qpos_indices(m)
    rad = jm.agent_deg_to_mjcf_rad(np.array([jm.STAND_AGENT_DEG[i] for i in jm.POLICY_JOINTS]))
    assert np.allclose(rad, m.key_qpos[0][qpos_idx], atol=0.1)
    back = jm.mjcf_rad_to_agent_deg(rad)
    assert np.allclose(back, [jm.STAND_AGENT_DEG[i] for i in jm.POLICY_JOINTS])
    assert (jm.ctrl_to_policy(jm.policy_to_ctrl(np.arange(8))) == np.arange(8)).all()


def test_model_report_present():
    r = json.loads((GEN / "model_report.json").read_text())
    assert set(r["variants"]) == set(VARIANTS)


def test_terrain_variants_settle_identically_to_flat():
    """Parked boxes (z=-1, 1 mm) cannot touch the robot: base_height_target stays valid for all four."""
    r = json.loads((GEN / "model_report.json").read_text())
    for engine in ("cpu", "gpu"):
        assert abs(r["variants"][engine]["settled_z"] - r["variants"][f"{engine}_terrain"]["settled_z"]) < 1e-5


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
def test_terrain_variant_has_parked_boxes_in_one_body_and_body_keyed_sensors(engine):
    from trainer.env import terrain as tr

    m = mujoco.MjModel.from_xml_path(str(GEN / f"bittle_{engine}_terrain.xml"))
    ids = tr.box_geom_ids(m)
    assert len(ids) == tr.MAX_BOXES
    terrain_body = m.body("terrain").id
    assert all(m.geom_bodyid[g] == terrain_body for g in ids) and m.geom_bodyid[m.geom("floor").id] == terrain_body
    assert (m.geom_pos[ids][:, 2] < -0.5).all()
    for i in range(m.nsensor):
        if m.sensor(i).name.endswith("_floor_found"):
            assert m.sensor_reftype[i] == mujoco.mjtObj.mjOBJ_BODY and m.sensor_refid[i] == terrain_body
    names = {m.sensor(i).name for i in range(m.nsensor)}
    assert {f"{leg}_shank_floor_found" for leg in jm.LEGS} <= names
    assert {f"{leg}_thigh_floor_found" for leg in jm.LEGS} <= names


def test_flat_variants_have_no_stumble_sensors():
    for engine in ("cpu", "gpu"):
        m = mujoco.MjModel.from_xml_path(str(GEN / f"bittle_{engine}.xml"))
        assert not any("shank" in m.sensor(i).name for i in range(m.nsensor))


def test_mesh_variant_excludes_chassis_vs_tucked_leg_contacts():
    """The firmware wkF/crF gaits (which the real robot walks) must not self-collide in the mesh model:
    MuJoCo collides the CONVEX HULL of the chassis meshes, which fills the underside where legs tuck."""
    from trainer.assets.opencat_gaits import GAITS, frame_to_mjcf_ctrl

    m = mujoco.MjModel.from_xml_path(str(GEN / "bittle_cpu.xml"))
    d = mujoco.MjData(m)
    assert m.nexclude == len(build_models.CHASSIS_BODIES) * len(build_models.LEG_TUCK_BODIES)
    for row in GAITS["wkF"] + GAITS["crF"]:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[2] = 0.3  # in the air: only self-contacts can exist
        ctrl = frame_to_mjcf_ctrl(row)
        for a in range(m.nu):
            d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] = ctrl[a]
        mujoco.mj_forward(m, d)
        assert not any(c.dist < 0 for c in d.contact[: d.ncon])
