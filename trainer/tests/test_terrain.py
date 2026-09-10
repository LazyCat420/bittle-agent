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
    # the spawn sits inside every parked box's xy footprint: their sentinel depth must not leak out
    for x, y in ((0.0, 0.0), (0.05, -0.05), (0.14, 0.14)):
        assert tr.terrain_height(np, x, y, f.box_pos, f.box_half, f.box_yaw) == 0.0
        assert np.all(tr.height_scan(np, x, y, 0.3, f.box_pos, f.box_half, f.box_yaw) == 0.0)
    feet = np.array([[0.05, 0.03, tr.PLANE_Z + tr.FOOT_RADIUS]] * 4)
    assert np.allclose(tr.foot_clearance(np, feet, f.box_pos, f.box_half, f.box_yaw), 0.0)
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


def test_shares_make_a_per_env_mixture_and_leave_the_full_field_stream_alone():
    """slope_share / box_share < 1 switch the slope / boxes off for a share of envs; the share draws
    come LAST, so an env that keeps its terrain gets exactly the field it would have had at share 1."""
    full = TerrainConfig(kind="rough_slope", n_boxes=10, slope_deg=(4.0, 10.0), slope_yaw_deg=(-180.0, 180.0),
                         box_height_m=(0.005, 0.015))
    mixed = full.model_copy(update={"slope_share": 0.5, "box_share": 0.5})
    n_slope = n_boxes = 0
    for seed in range(400):
        a = tr.sample_field_numpy(np.random.default_rng(seed), full)
        b = tr.sample_field_numpy(np.random.default_rng(seed), mixed)
        has_slope = np.degrees(tr.slope_from_gravity(np, b.gravity)[0]) > 1e-9
        n_slope += has_slope
        n_boxes += b.n_boxes > 0
        assert b.n_boxes in (0, 10)
        if has_slope:
            assert np.allclose(a.gravity, b.gravity)
        else:
            assert np.allclose(b.gravity, [0.0, 0.0, -tr.G])
        if b.n_boxes:
            assert np.allclose(a.box_pos, b.box_pos) and np.allclose(a.box_yaw, b.box_yaw)
        else:
            assert (b.box_pos[:, 2] < -0.5).all()
    assert 150 < n_slope < 250 and 150 < n_boxes < 250
    off = tr.sample_field_numpy(np.random.default_rng(3), full.model_copy(update={"slope_share": 0.0, "box_share": 0.0}))
    assert off.n_boxes == 0 and np.allclose(off.gravity, [0.0, 0.0, -tr.G])


def test_house_level_is_a_mixture_over_every_kind():
    """terrain.level 4: a batch of envs contains flat, slope-only, rocks-only and both."""
    cfg = apply_patch(None, {"terrain": {"level": 4}})
    t = cfg.terrain
    assert t.kind == "rough_slope" and t.slope_share == 0.6 and t.box_share == 0.6
    assert tuple(t.slope_yaw_deg) == (-180.0, 180.0) and t.slope_deg[1] <= 10.0 and t.box_height_m[1] <= 0.015
    kinds = set()
    for seed in range(200):
        f = tr.sample_field_numpy(np.random.default_rng(seed), t)
        sloped = np.degrees(tr.slope_from_gravity(np, f.gravity)[0]) > 1e-9
        kinds.add((bool(sloped), f.n_boxes > 0))
    assert kinds == {(False, False), (True, False), (False, True), (True, True)}
    # downhill happens: some sampled gravities pull FORWARD (+x)
    assert any(tr.sample_field_numpy(np.random.default_rng(s), t).gravity[0] > 0.5 for s in range(200))


def test_box_share_zero_env_reads_flat_ground_at_the_spawn():
    """The GPU-side symptom of the parked-box leak: a level-4 env that drew no boxes must see
    terrain_h == 0 under the torso and a zero base_height error at the stand, exactly like flat ground."""
    cfg = apply_patch(None, {"terrain": {"level": 4, "box_share": 0.0, "slope_share": 0.0}, "dr": {"enabled": False}})
    env = BittleCpuEnv(cfg, seed=0)
    env.reset(command=np.array([0.0, 0.0, 0.0]))
    assert env.model_params.terrain.n_boxes == 0
    for _ in range(25):
        obs, r, done, info = env.step(np.zeros(8))
    assert info.terrain_h == 0.0 and abs(info.terms["base_height"]) < 0.05, info.terms["base_height"]
    assert np.all(np.abs(info.foot_clearance) < 0.005)


def test_foot_clearance_term_is_order_one_for_a_shuffling_swing():
    """A swing foot that never lifts (clearance 0 vs the 12 mm target) must cost O(0.1) per foot, the same
    mm^2/1000 convention as base_height — unscaled it was 1.4e-4 and invisible at any allowed weight."""
    q = {"up_world": np.array([0.0, 0.0, 1.0]), "terrain_h": 0.0, "torque_cap": np.full(8, 0.25),
         "foot_clearance": np.zeros(4), "limb_contact": np.zeros(8), "uphill_xy": np.zeros(2), "cmd": np.array([0.1, 0.0, 0.0]),
         "local_linvel": np.array([0.1, 0.0, 0.0]), "gyro": np.zeros(3), "global_linvel": np.array([0.1, 0.0, 0.0]),
         "global_angvel": np.zeros(3), "gravity": np.array([0.0, 0.0, -1.0]), "up_z": 1.0, "torso_z": 0.048,
         "action": np.zeros(8), "last_action": np.zeros(8), "torques": np.zeros(8), "joint_vel": np.zeros(8),
         "target_norm": np.zeros(8), "target_deg": spec.STAND_DEG.copy(), "feet_air_time": np.full(4, 0.1),
         "first_contact": np.zeros(4), "contact": np.array([0.0, 1.0, 1.0, 0.0]), "feet_vel_xy": np.full((4, 2), 0.05)}
    terms = spec.reward_terms(np, q, 0.01, 0.25, 0.048)
    per_foot = 1000.0 * spec.FOOT_CLEARANCE_TARGET ** 2
    assert abs(terms["foot_clearance"] - 2 * per_foot) < 1e-9 and 0.1 < per_foot < 0.2  # two feet in swing


def test_foot_clearance_term_is_one_sided():
    """Clearing the 12 mm target by a margin costs nothing; falling short costs the squared shortfall
    (mm^2/1000). The gate is a per-swing PEAK >= 8 mm, so overshoot must never be penalised."""
    base = {"up_world": np.array([0.0, 0.0, 1.0]), "terrain_h": 0.0, "torque_cap": np.full(8, 0.25),
            "limb_contact": np.zeros(8), "uphill_xy": np.zeros(2), "cmd": np.array([0.1, 0.0, 0.0]),
            "local_linvel": np.array([0.1, 0.0, 0.0]), "gyro": np.zeros(3), "global_linvel": np.array([0.1, 0.0, 0.0]),
            "global_angvel": np.zeros(3), "gravity": np.array([0.0, 0.0, -1.0]), "up_z": 1.0, "torso_z": 0.048,
            "action": np.zeros(8), "last_action": np.zeros(8), "torques": np.zeros(8), "joint_vel": np.zeros(8),
            "target_norm": np.zeros(8), "target_deg": spec.STAND_DEG.copy(), "feet_air_time": np.full(4, 0.1),
            "first_contact": np.zeros(4), "contact": np.array([0.0, 1.0, 1.0, 0.0]), "feet_vel_xy": np.full((4, 2), 0.05)}
    high = spec.reward_terms(np, dict(base, foot_clearance=np.full(4, 0.020)), 0.01, 0.25, 0.048)["foot_clearance"]
    exact = spec.reward_terms(np, dict(base, foot_clearance=np.full(4, 0.012)), 0.01, 0.25, 0.048)["foot_clearance"]
    low = spec.reward_terms(np, dict(base, foot_clearance=np.full(4, 0.002)), 0.01, 0.25, 0.048)["foot_clearance"]
    assert high == 0.0 and exact == 0.0
    assert abs(low - 2 * 1000.0 * 0.010 ** 2) < 1e-9
