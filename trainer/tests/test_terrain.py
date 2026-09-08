"""trainer/env/terrain.py — the shared analytic terrain, and the flat-ground no-op guarantee."""

import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from trainer.config import TrainConfig, TerrainConfig, apply_patch  # noqa: E402
from trainer.env import spec, terrain as tr  # noqa: E402
from trainer.env.cpu_env import BittleCpuEnv  # noqa: E402

HERE = Path(__file__).resolve().parent


def test_variant_for_and_kinds():
    assert tr.variant_for("flat", "cpu") == "cpu" and tr.variant_for("slope", "gpu") == "gpu"
    assert tr.variant_for("rough", "cpu") == "cpu_terrain" and tr.variant_for("rough_slope", "gpu") == "gpu_terrain"
    with pytest.raises(ValueError):
        tr.variant_for("hfield", "cpu")


def test_gravity_roundtrip_and_norm():
    for slope in np.radians([0.0, 3.0, 8.0, 14.0, 20.0]):
        for yaw in np.radians([-170.0, -45.0, 0.0, 30.0, 180.0]):
            g = tr.gravity_for_slope(np, slope, yaw)
            assert abs(np.linalg.norm(g) - tr.G) < 1e-9
            s2, y2 = tr.slope_from_gravity(np, g)
            assert abs(s2 - slope) < 1e-9
            if slope > 0:
                assert abs((y2 - yaw + np.pi) % (2 * np.pi) - np.pi) < 1e-9
    # walking +x uphill (yaw 0): gravity pulls back along -x
    g = tr.gravity_for_slope(np, np.radians(8.0), 0.0)
    assert g[0] < 0 and abs(g[1]) < 1e-12 and tr.uphill_xy(np, g)[0] > 0
    assert np.allclose(tr.uphill_xy(np, np.array([0.0, 0.0, -tr.G])), 0.0)


def test_terrain_height_zero_when_parked_or_no_boxes():
    f = tr.flat_field()
    assert tr.terrain_height(np, 0.3, 0.1, f.box_pos, f.box_half, f.box_yaw) == 0.0
    empty = np.zeros((0, 3))
    assert tr.terrain_height(np, 0.3, 0.1, empty, empty, np.zeros(0)) == 0.0


def test_sample_field_respects_config_and_parks_the_rest():
    tcfg = TerrainConfig(kind="rough_slope", n_boxes=5, slope_deg=(6.0, 6.0), box_height_m=(0.01, 0.01),
                         box_size_m=(0.03, 0.03), field_start_m=0.2, field_width_m=0.4)
    f = tr.sample_field_numpy(np.random.default_rng(1), tcfg)
    assert f.n_boxes == 5 and f.box_pos.shape == (tr.MAX_BOXES, 3)
    assert abs(np.degrees(tr.slope_from_gravity(np, f.gravity)[0]) - 6.0) < 1e-9
    live = f.box_pos[:5]
    assert (live[:, 0] > 0.1).all() and (np.abs(live[:, 1]) <= 0.2).all()
    assert np.allclose(live[:, 2] + f.box_half[:5, 2] - tr.PLANE_Z, 0.01)  # box tops protrude 10 mm
    assert (f.box_pos[5:, 2] < -0.5).all()


def test_analytic_height_matches_mujoco_ray():
    """The load-bearing claim: terrain_height is EXACT against the compiled model (yaw-only boxes)."""
    env = BittleCpuEnv(apply_patch(None, {"terrain": {"level": 3}}), seed=3)
    env.reset()
    m, d = env.model, env.data
    rng = np.random.default_rng(0)
    geomid = np.zeros(1, dtype=np.int32)
    on_boxes = 0
    for _ in range(500):
        x, y = rng.uniform(0.3, 2.5), rng.uniform(-0.3, 0.3)  # x >= 0.3 keeps the ray off the robot
        h = env._terrain_h(x, y)
        dist = mujoco.mj_ray(m, d, np.array([x, y, 0.5]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        assert abs((0.5 - dist - tr.PLANE_Z) - h) < 1e-6
        on_boxes += h > 0
    assert on_boxes > 20


def test_orientation_identity_on_flat_ground():
    """sum(gravity_body[:2]^2) == up_world[:2]^2 for every rotation: the up-vector form of the
    orientation term is bit-for-bit the old term on flat ground, and correct on a slope."""
    rng = np.random.default_rng(0)
    for _ in range(1000):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        gravity_body = spec.gravity_from_xmat(np, R)
        up_world = R[:, 2]
        assert abs(np.sum(gravity_body[:2] ** 2) - np.sum(up_world[:2] ** 2)) < 1e-12


def test_flat_terrain_is_a_noop_against_pre_terrain_golden():
    """Golden values recorded from the PRE-terrain trainer (HEAD 92f2ccb) on the flat env.

    The fixture pins the OLD generator on purpose (a-digest-fixture-pins-the-generator): it
    must never be regenerated from this code, or the guarantee it carries — that terrain
    support changed nothing on flat ground — becomes a tautology.
    """
    g = json.loads((HERE / "golden_flat_92f2ccb.json").read_text())
    env = BittleCpuEnv(TrainConfig(dr={"enabled": False}), seed=0)
    env.reset(command=np.array([0.12, 0.0, 0.0]))
    for a, row in zip(g["actions"], g["rows"]):
        obs, r, done, info = env.step(np.array(a))
        assert abs(r - row["reward"]) < 1e-9 and done == row["done"]
        assert np.allclose(obs, row["obs"], atol=1e-9)
        for k, v in row["terms"].items():
            assert abs(info.terms[k] - v) < 1e-9, k
        for k in ("foot_clearance", "stumble", "slope_progress", "stall"):
            assert k in info.terms
        assert info.terms["stumble"] == 0.0 and info.terms["slope_progress"] == 0.0 and info.terrain_h == 0.0
