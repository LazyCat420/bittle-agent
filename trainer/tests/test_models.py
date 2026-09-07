import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from trainer.assets import build_models, joint_map as jm  # noqa: E402

GEN = Path(build_models.OUT_DIR)


@pytest.mark.parametrize("variant", ["cpu", "gpu"])
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
    for name in ("bittle_cpu.xml", "bittle_gpu.xml"):
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
    assert set(r["variants"]) == {"cpu", "gpu"}
