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
         "feet_stance_time": np.zeros(4), "support_rise": 0.0,
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
         "feet_stance_time": np.zeros(4), "support_rise": 0.0,
            "first_contact": np.zeros(4), "contact": np.array([0.0, 1.0, 1.0, 0.0]), "feet_vel_xy": np.full((4, 2), 0.05)}
    high = spec.reward_terms(np, dict(base, foot_clearance=np.full(4, 0.020)), 0.01, 0.25, 0.048)["foot_clearance"]
    exact = spec.reward_terms(np, dict(base, foot_clearance=np.full(4, 0.012)), 0.01, 0.25, 0.048)["foot_clearance"]
    low = spec.reward_terms(np, dict(base, foot_clearance=np.full(4, 0.002)), 0.01, 0.25, 0.048)["foot_clearance"]
    assert high == 0.0 and exact == 0.0
    assert abs(low - 2 * 1000.0 * 0.010 ** 2) < 1e-9


def test_field_to_json_carries_the_enabled_boxes_and_the_incline():
    from trainer.eval.gates import load_suite

    proto = load_suite("rough_v1")["protocol"]["terrain"]
    f = tr.field_from_protocol(proto)
    j = tr.field_to_json(f)
    assert len(j["boxes"]) == proto["n_boxes"] and j["slope_deg"] == 0.0 and j["plane_z"] == tr.PLANE_Z
    b = j["boxes"][0]
    assert abs((b["pos"][2] + b["half"][2]) - (tr.PLANE_Z + proto["box_height_m"])) < 1e-6  # the top is 12 mm up
    slope = tr.field_from_protocol({"kind": "slope", "slope_deg": 8.0, "slope_yaw_deg": 0.0})
    js = tr.field_to_json(slope)
    assert abs(js["slope_deg"] - 8.0) < 1e-3 and js["boxes"] == []
    assert tr.field_to_json(tr.flat_field())["boxes"] == []


# ── stairs (2026-09-10) ──────────────────────────────────────────────────────

def _stairs(n=3, rise=0.018, tread=0.08, profile="up_down", seed=0, **kw):
    tcfg = TerrainConfig(kind="stairs", stair_steps=(n, n), stair_rise_m=(rise, rise), stair_tread_m=(tread, tread),
                         stair_profile=profile, **kw)
    return tr.sample_field_numpy(np.random.default_rng(seed), tcfg)


def test_stairs_geometry_up_landing_down():
    """3 x 18 mm up, a 0.30 m landing, 3 down: tops rise by one riser per tread, every box bottom is buried,
    nothing exceeds the baked parked size, the rest of the 32 slots stay parked. The last step DOWN is the
    floor itself, so a flight of n needs n - 1 descent boxes: 2n live boxes in all."""
    f = _stairs()
    live = f.box_pos[:, 2] > -0.5
    assert live.sum() == 6 and f.n_boxes == 6
    tops = f.box_pos[:, 2] + f.box_half[:, 2] - tr.PLANE_Z
    bottoms = f.box_pos[:, 2] - f.box_half[:, 2]
    assert np.allclose(tops[:3], [0.018, 0.036, 0.054])                       # flight up
    assert tops[tr.STAIR_LANDING_SLOT] == pytest.approx(0.054)                 # landing at the top
    assert np.allclose(tops[tr.STAIR_LANDING_SLOT + 1: tr.STAIR_LANDING_SLOT + 3], [0.036, 0.018])
    assert not live[tr.STAIR_LANDING_SLOT + 3]                                 # the floor is the last step down
    assert np.allclose(bottoms[live], tr.PLANE_Z - tr.STAIR_BURY)              # no gap under any tread
    assert (f.box_half[live] <= tr.PARKED_HALF + 1e-9).all()                   # never grows past the compiled size
    # x layout: risers every 80 mm from field_start, the landing right after the flight, the descent after it
    xs = f.box_pos[:, 0]
    assert np.allclose(xs[:3], 0.15 + np.array([0.04, 0.12, 0.20]))
    assert xs[tr.STAIR_LANDING_SLOT] == pytest.approx(0.15 + 0.24 + 0.15)
    assert xs[tr.STAIR_LANDING_SLOT + 1] == pytest.approx(0.15 + 0.24 + 0.30 + 0.04)
    assert (f.box_yaw == 0).all() and np.allclose(f.gravity, [0, 0, -tr.G])


def test_stairs_height_profile_is_a_staircase():
    """terrain_height along the centre line reads 0, 18, 36, 54 (landing), 36, 18, 0 mm."""
    f = _stairs()
    h = lambda x: tr.terrain_height(np, x, 0.0, f.box_pos, f.box_half, f.box_yaw)
    assert h(0.0) == 0.0 and h(0.14) == 0.0
    assert h(0.19) == pytest.approx(0.018) and h(0.27) == pytest.approx(0.036) and h(0.35) == pytest.approx(0.054)
    assert h(0.55) == pytest.approx(0.054)                       # on the landing
    assert h(0.73) == pytest.approx(0.036) and h(0.81) == pytest.approx(0.018)
    assert h(0.87) == 0.0 and h(0.95) == 0.0                     # the third step down is the floor
    assert h(0.35, ) == h(0.35) and tr.terrain_height(np, 0.35, 0.29, f.box_pos, f.box_half, f.box_yaw) == pytest.approx(0.054)
    assert tr.terrain_height(np, 0.35, 0.31, f.box_pos, f.box_half, f.box_yaw) == 0.0   # 0.60 m wide


def test_stairs_profiles_up_only_and_down_platform():
    up = _stairs(profile="up")
    assert up.n_boxes == 4 and (up.box_pos[tr.STAIR_LANDING_SLOT + 1:, 2] < -0.5).all()
    down = _stairs(profile="down")
    assert down.n_boxes == 3 and (down.box_pos[:tr.STAIR_LANDING_SLOT, 2] < -0.5).all()   # platform + 2 steps
    # the platform sits BEHIND field_start, under the spawn, at the top level; the flight descends from field_start
    h = lambda x: tr.terrain_height(np, x, 0.0, down.box_pos, down.box_half, down.box_yaw)
    assert h(0.0) == pytest.approx(0.054) and h(-0.14) == pytest.approx(0.054) and h(-0.16) == 0.0
    assert h(0.19) == pytest.approx(0.036) and h(0.27) == pytest.approx(0.018) and h(0.35) == pytest.approx(0.0)


def test_stairs_sampled_ranges_and_box_share():
    tcfg = TerrainConfig(kind="stairs", stair_steps=(1, 4), stair_rise_m=(0.006, 0.022), stair_tread_m=(0.06, 0.12))
    ns, rises = set(), []
    for seed in range(60):
        f = tr.sample_field_numpy(np.random.default_rng(seed), tcfg)
        n = int((f.box_pos[:tr.STAIR_LANDING_SLOT, 2] > -0.5).sum())
        ns.add(n)
        assert f.n_boxes == 2 * n
        rises.append(f.box_pos[0, 2] + f.box_half[0, 2] - tr.PLANE_Z)
    assert ns == {1, 2, 3, 4}
    assert min(rises) >= 0.006 - 1e-9 and max(rises) <= 0.022 + 1e-9 and max(rises) - min(rises) > 0.008
    mixed = TerrainConfig(kind="stairs", box_share=0.5)
    counts = [tr.sample_field_numpy(np.random.default_rng(s), mixed).n_boxes for s in range(40)]
    assert 0 in counts and 6 in counts and set(counts) <= {0, 6}


def test_support_height_is_the_mean_under_the_feet():
    f = _stairs()
    # front pair on step 1 (18 mm), rear pair on the runway: half a riser
    feet = np.array([[0.19, 0.047, 0.05], [0.19, -0.047, 0.05], [0.085, 0.047, 0.03], [0.085, -0.047, 0.03]])
    assert tr.support_height(np, feet, f.box_pos, f.box_half, f.box_yaw) == pytest.approx(0.009)
    # all four on the landing: the full height; all on the runway: 0; parked field: 0 anywhere
    assert tr.support_height(np, feet + [0.35, 0, 0], f.box_pos, f.box_half, f.box_yaw) == pytest.approx(0.054)
    assert tr.support_height(np, feet - [0.10, 0, 0], f.box_pos, f.box_half, f.box_yaw) == 0.0
    flat = tr.flat_field()
    assert tr.support_height(np, feet, flat.box_pos, flat.box_half, flat.box_yaw) == 0.0


def test_stairs_protocol_field_is_deterministic_and_draws_in_the_viewer():
    proto = {"kind": "stairs", "stair_steps": 3, "stair_rise_m": 0.018, "stair_tread_m": 0.08,
             "stair_profile": "up_down", "field_start_m": 0.15}
    a, b = tr.field_from_protocol(proto), tr.field_from_protocol(proto)
    assert np.array_equal(a.box_pos, b.box_pos) and a.n_boxes == 6
    js = tr.field_to_json(a)
    assert len(js["boxes"]) == 6 and js["slope_deg"] == 0.0
    tops = sorted(round(bx["pos"][2] + bx["half"][2] - js["plane_z"], 4) for bx in js["boxes"])
    assert tops == [0.018, 0.018, 0.036, 0.036, 0.054, 0.054]
    assert all(bx["half"][1] == 0.3 for bx in js["boxes"])


def test_stairs_variant_and_kind_registry():
    assert tr.has_boxes("stairs") and tr.has_stairs("stairs") and not tr.has_slope("stairs")
    assert tr.variant_for("stairs", "cpu") == "cpu_terrain" and tr.variant_for("stairs", "gpu") == "gpu_terrain"
    assert "stairs" in tr.KINDS and 2 * tr.MAX_STAIR_STEPS + 1 <= tr.MAX_BOXES


# ── stance_timeout: the swing term that pays before a swing exists (2026-09-10) ──

def _q_for_stance(stance, contact, cmd=(0.12, 0.0, 0.0)):
    return {"cmd": np.array(cmd), "local_linvel": np.zeros(3), "gyro": np.zeros(3), "global_linvel": np.zeros(3),
            "global_angvel": np.zeros(3), "up_world": np.array([0.0, 0.0, 1.0]), "torso_z": 0.047, "terrain_h": 0.0,
            "action": np.zeros(8), "last_action": np.zeros(8), "torques": np.zeros(8), "joint_vel": np.zeros(8),
            "torque_cap": np.ones(8), "target_norm": np.zeros(8), "target_deg": spec.STAND_DEG.copy(),
            "feet_air_time": np.zeros(4), "feet_stance_time": np.asarray(stance, dtype=float), "support_rise": 0.0,
            "first_contact": np.zeros(4), "contact": np.asarray(contact, dtype=float),
            "feet_vel_xy": np.zeros((4, 2)), "foot_clearance": np.zeros(4), "limb_contact": np.zeros(8),
            "uphill_xy": np.zeros(2)}


def test_stance_timeout_grows_while_a_foot_drags_and_is_zero_for_a_healthy_gait():
    """The term the dragging gait needed: it pays BEFORE a swing exists. A trot-like gait (every foot
    planted less than STANCE_MAX_S) earns exactly 0, a robot with all four feet planted for a second
    earns 4 x (1.0 - 0.3), and lifting a foot zeroes that foot's contribution immediately."""
    t = spec.STANCE_MAX_S
    healthy = spec.reward_terms(np, _q_for_stance([0.1, 0.2, 0.0, 0.15], [1, 1, 0, 1]), 0.25, 0.25, 0.047)
    assert healthy["stance_timeout"] == 0.0
    dragging = spec.reward_terms(np, _q_for_stance([0.6] * 4, [1] * 4), 0.25, 0.25, 0.047)
    assert dragging["stance_timeout"] == pytest.approx(4 * (0.6 - t))
    # exactly at the threshold: still free; one foot lifts (its clock reset by the env): its share goes
    at = spec.reward_terms(np, _q_for_stance([t] * 4, [1] * 4), 0.25, 0.25, 0.047)
    assert at["stance_timeout"] == 0.0
    lifted = spec.reward_terms(np, _q_for_stance([0.6, 0.6, 0.0, 0.6], [1, 1, 0, 1]), 0.25, 0.25, 0.047)
    assert lifted["stance_timeout"] == pytest.approx(3 * (0.6 - t))
    # a ZERO command must never ask the robot to pick its feet up (statue task)
    still = spec.reward_terms(np, _q_for_stance([9.0] * 4, [1] * 4, cmd=(0.0, 0.0, 0.0)), 0.25, 0.25, 0.047)
    assert still["stance_timeout"] == 0.0


def test_stance_timeout_is_off_by_default_and_bounded():
    from trainer.config import RewardWeights, TrainConfig

    assert RewardWeights().stance_timeout == 0.0
    assert TrainConfig().reward.weights.stance_timeout == 0.0
    f = RewardWeights.model_fields["stance_timeout"]
    assert [m for m in f.metadata if getattr(m, "le", None) == 0.0] and [m for m in f.metadata if getattr(m, "ge", None) == -10.0]


def test_the_cpu_env_stance_clock_counts_contact_and_resets_on_lift():
    """The clock the term reads is real bookkeeping, not a constant: standing still it rises on every
    foot; the air-time and stance clocks are complementary (a foot is either planted or flying)."""
    from trainer.config import apply_patch

    env = BittleCpuEnv(apply_patch(None, {"dr": {"enabled": False}}), envelope_tier="tested", seed=0)
    env.reset(command=np.array([0.0, 0.0, 0.0]))
    for _ in range(40):
        _, _, _, info = env.step(np.zeros(8))
    assert (env.feet_stance_time > 0.5).all()                      # 0.8 s of standing
    assert np.allclose(env.feet_air_time[env.feet_stance_time > 0], 0.0)
    planted = env.feet_stance_time.copy()
    # drive the left-front leg up hard: its stance clock must reset while the others keep counting
    act = np.zeros(8)
    act[0], act[4] = 1.0, 1.0
    for _ in range(25):
        env.step(act)
    assert env.feet_stance_time[1] < planted[1], (env.feet_stance_time, planted)


def test_stance_timeout_is_capped_so_it_cannot_flatten_the_reward_to_the_clip_floor():
    """weighted_reward clips the total at 0: an unbounded penalty would give a flat zero reward and NO
    gradient. The term saturates at 4 x STANCE_OVERDUE_CAP_S however long the feet stay down, so at the
    intended weight the step reward stays positive next to a typical tracking reward."""
    t, cap = spec.STANCE_MAX_S, spec.STANCE_OVERDUE_CAP_S
    for stance in (t + cap, 1.0, 9.0, 60.0):
        terms = spec.reward_terms(np, _q_for_stance([stance] * 4, [1] * 4), 0.25, 0.25, 0.047)
        assert terms["stance_timeout"] == pytest.approx(4 * cap)
    worst = spec.reward_terms(np, _q_for_stance([9.0] * 4, [1] * 4), 0.25, 0.25, 0.047)
    # a dragging robot still tracking badly: tracking 1.5 x ~0.6 against the capped penalty at -0.1
    total = spec.weighted_reward(np, worst, {"stance_timeout": -0.1, "tracking_lin_vel": 1.5}, 0.02)
    assert 0.0 < float(total) < 1.5, float(total)
    assert float(spec.weighted_reward(np, worst, {"stance_timeout": -0.1}, 0.02)) == 0.0  # alone it clips


def test_climb_progress_pays_for_a_staircase_and_is_zero_on_flat_and_slope():
    """slope_progress reads the gravity tilt, which is exactly 0 on stairs -- a staircase is level ground
    at several heights -- so a climb earns nothing from it. climb_progress is its stairs twin: the RATE
    the support surface under the feet rises, positive only, and only while commanded to move."""
    rise = 0.018
    q = _q_for_stance([0.1] * 4, [1] * 4)
    up = spec.reward_terms(np, dict(q, support_rise=rise), 0.25, 0.25, 0.047, dt_ref=0.02)
    assert up["climb_progress"] == pytest.approx(rise / 0.02 * spec.CLIMB_RATE_SCALE)   # 0.9 m/s, scaled
    assert up["slope_progress"] == 0.0                              # the stairs are level: no gravity tilt
    # going DOWN pays nothing (it must never be cheaper to fall off the flight than to walk it)
    down = spec.reward_terms(np, dict(q, support_rise=-rise), 0.25, 0.25, 0.047, dt_ref=0.02)
    assert down["climb_progress"] == 0.0
    # flat ground and a standing robot are exact no-ops
    assert spec.reward_terms(np, q, 0.25, 0.25, 0.047)["climb_progress"] == 0.0
    still = _q_for_stance([0.1] * 4, [1] * 4, cmd=(0.0, 0.0, 0.0))
    assert spec.reward_terms(np, dict(still, support_rise=rise), 0.25, 0.25, 0.047)["climb_progress"] == 0.0
    # the rate is per SECOND: the same riser crossed at 25 Hz scores half as much per step
    slow = spec.reward_terms(np, dict(q, support_rise=rise), 0.25, 0.25, 0.047, dt_ref=0.04)
    assert slow["climb_progress"] == pytest.approx(up["climb_progress"] / 2)


def test_climb_progress_is_off_by_default_and_the_cpu_env_measures_a_real_step_up():
    """Default weight 0 keeps every existing config unchanged; and the support_rise the term reads is real
    bookkeeping: a robot lifted onto a stair platform between two steps reports a positive rise once."""
    from trainer.config import RewardWeights, apply_patch
    from trainer.env.cpu_env import BittleCpuEnv

    assert RewardWeights().climb_progress == 0.0
    env = BittleCpuEnv(apply_patch(None, {"terrain": {"level": 5, "stair_steps": [3, 3], "stair_rise_m": [0.018, 0.018],
                                                      "stair_profile": "up", "spawn_jitter_m": 0.0},
                                          "dr": {"enabled": False}}), envelope_tier="tested", seed=0)
    env.reset(command=np.array([0.12, 0.0, 0.0]))
    rises = []
    for _ in range(5):
        _, _, _, info = env.step(np.zeros(8))
        rises.append(info.terms["climb_progress"])
    assert all(r == 0.0 for r in rises)                              # standing on the runway: no rise
    env.data.qpos[0] += 0.19                                          # teleport the whole robot onto step 1
    env.data.qpos[2] += 0.018
    _, _, _, info = env.step(np.zeros(8))
    assert info.terms["climb_progress"] > 0.0, "the support surface rose but the term paid nothing"
    _, _, _, info2 = env.step(np.zeros(8))
    assert info2.terms["climb_progress"] == 0.0                       # it pays for the RISE, not for height


def test_climb_progress_scale_lets_an_allowed_weight_buy_a_real_reward_share():
    """The scale is a CALIBRATION, so pin what it buys rather than the constant. Climbing a whole
    stairs_v1 flight (3 x 18 mm) over one episode must be worth roughly a tenth of a typical episode's
    reward at a mid-range weight -- unscaled it was 2.8 % at weight 5 and could not exceed 5.4 % at the
    bound, which is the same dead zone the foot_clearance rescale was written to escape."""
    dt, flight, typical = 0.02, 3 * 0.018, 470.0
    q = _q_for_stance([0.1] * 4, [1] * 4)
    per_step = spec.reward_terms(np, dict(q, support_rise=0.018), 0.25, 0.25, 0.047, dt_ref=dt)["climb_progress"]
    episode_sum = per_step * (flight / 0.018)          # the whole flight, one riser per control step
    for weight, lo, hi in ((2.0, 0.07, 0.20), (10.0, 0.25, 0.60)):
        share = episode_sum * weight / (typical + episode_sum * weight)
        assert lo < share < hi, (weight, share)
    # and a weight inside the bound must still be able to stay small: the term is linear in the weight
    assert spec.reward_terms(np, dict(q, support_rise=0.018), 0.25, 0.25, 0.047)["climb_progress"] > 0
