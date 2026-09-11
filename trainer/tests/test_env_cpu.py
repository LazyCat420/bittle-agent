import numpy as np
import pytest

pytest.importorskip("mujoco")

from trainer.config import TrainConfig  # noqa: E402
from trainer.env import spec  # noqa: E402
from trainer.env.cpu_env import BittleCpuEnv  # noqa: E402
from trainer.env.dr import nominal_episode_params, nominal_model_params  # noqa: E402


def make(seed=0, **patch):
    return BittleCpuEnv(TrainConfig(**patch), seed=seed)


def test_obs_shape_and_layout():
    env = make()
    obs = env.reset(command=np.zeros(3))
    assert obs.shape == (spec.obs_size(3, False),) == (41,) and np.isfinite(obs).all()
    env2 = BittleCpuEnv(TrainConfig(obs={"history_n": 5, "phase_clock": True}))
    assert env2.reset().shape == (spec.obs_size(5, True),)


def test_zero_action_stand_holds_two_seconds():
    env = BittleCpuEnv(TrainConfig(dr={"enabled": False}), seed=1)
    env.reset(command=np.zeros(3))
    for _ in range(100):
        _, r, done, info = env.step(np.zeros(8))
        assert not done
    assert 0.044 < env.data.qpos[2] < 0.050
    assert info.contact.sum() == 4 and r > 0


def test_determinism():
    a, b = make(seed=7), make(seed=7)
    oa, ob = a.reset(), b.reset()
    acts = np.random.default_rng(0).uniform(-1, 1, (40, 8))
    for k in range(40):
        oa, _, _, _ = a.step(acts[k])
        ob, _, _, _ = b.step(acts[k])
    assert np.array_equal(oa, ob)


def test_tracking_term_prefers_matching_velocity():
    env = BittleCpuEnv(TrainConfig(dr={"enabled": False}))
    env.reset(command=np.array([0.12, 0, 0]))
    _, _, _, info = env.step(np.zeros(8))
    standing = info.terms["tracking_lin_vel"]
    env.reset(command=np.zeros(3))
    _, _, _, info = env.step(np.zeros(8))
    assert info.terms["tracking_lin_vel"] > standing


def test_flip_terminates():
    env = BittleCpuEnv(TrainConfig(dr={"enabled": False}))
    env.reset()
    env.data.qpos[3:7] = [0, 1, 0, 0]  # upside down
    _, _, done, _ = env.step(np.zeros(8))
    assert done


def test_latency_buffer_delays_targets():
    ep = nominal_episode_params()
    ep.latency_steps = 3
    env = BittleCpuEnv(TrainConfig(dr={"enabled": False}), dr_override=nominal_model_params(), episode_override=ep)
    env.reset(command=np.zeros(3))
    t0 = env.target_deg.copy()
    applied = []
    for _ in range(4):
        _, _, _, info = env.step(np.ones(8))
        applied.append(info.applied_deg.copy())
    assert np.array_equal(applied[0], t0) and np.array_equal(applied[2], t0)
    assert not np.array_equal(applied[3], t0)


def test_targets_are_integer_degrees_within_envelope():
    env = make()
    env.reset()
    for _ in range(20):
        _, _, _, info = env.step(np.ones(8))
    assert np.array_equal(info.target_deg, np.round(info.target_deg))
    assert (info.target_deg <= env.hi).all() and (info.target_deg >= env.lo).all()


def test_target_override_replays_gait():
    from trainer.eval.evaluator import GaitController, StandController, run_episode

    env = BittleCpuEnv(TrainConfig(dr={"enabled": False}), envelope_tier="tested",
                       dr_override=nominal_model_params(), episode_override=nominal_episode_params())
    st = run_episode(env, StandController(), seed=0, seconds=2.0, command=np.array([0.0, 0, 0]))
    assert not st.fell and abs(st.distance_x) < 0.05
    st = run_episode(env, GaitController.from_opencat("trF"), seed=0, seconds=3.0, command=np.array([0.12, 0, 0]))
    assert st.steps > 0


# ── terrain ──────────────────────────────────────────────────────────────

def test_slope_uses_the_flat_xml_and_rotates_gravity():
    from trainer.config import apply_patch
    from trainer.env import terrain as tr

    env = BittleCpuEnv(apply_patch(None, {"terrain": {"level": 1, "slope_deg": [8.0, 8.0]}, "dr": {"enabled": False}}))
    env.reset(command=np.zeros(3))
    assert env.xml_path.name == "bittle_cpu.xml"
    g = env.model.opt.gravity
    assert abs(np.linalg.norm(g) - 9.81) < 1e-9 and abs(np.degrees(tr.slope_from_gravity(np, g)[0]) - 8.0) < 1e-9
    for _ in range(100):
        _, _, done, info = env.step(np.zeros(8))
        assert not done
    assert 0.044 < env.data.qpos[2] < 0.050  # height along the terrain normal, unchanged by the tilt
    assert info.terms["orientation"] < 0.02  # standing normal to the slope is NOT penalised


def test_rough_terrain_places_boxes_and_stumble_sensors_do_not_terminate():
    from trainer.config import apply_patch

    env = BittleCpuEnv(apply_patch(None, {"terrain": {"level": 2}, "dr": {"enabled": False}}), seed=2)
    env.reset(command=np.zeros(3))
    assert env.xml_path.name == "bittle_cpu_terrain.xml" and len(env.limb_found) == 8
    assert env.model_params.terrain.n_boxes == 24 and (env.model.geom_pos[env.box_ids][:24, 2] > -0.1).all()
    assert set(env.body_found) == {"torso_floor_found", "head__1_floor_found"}
    # force a shank onto the ground for 20 steps: a stumble, not a fall
    env.reset(command=np.zeros(3))
    env.data.qpos[2] = 0.005
    saw_limb = False
    for _ in range(20):
        _, _, done, info = env.step(np.zeros(8))
        saw_limb |= info.limb_contact.sum() > 0
        if done:
            break
    assert saw_limb and info.terms["stumble"] > 0


def test_walks_over_a_single_step_edge():
    """One 10 mm box across the runway: the firmware trot feels it (terrain_h under the torso rises)."""
    from trainer.config import apply_patch
    from trainer.env import terrain as tr
    from trainer.eval.evaluator import GaitController, run_episode

    field = tr.flat_field()
    field.box_pos[0] = [0.25, 0.0, tr.PLANE_Z + 0.010 - tr.BOX_HALF_Z]
    field.box_half[0] = [0.03, 0.30, tr.BOX_HALF_Z]
    cfg = apply_patch(None, {"terrain": {"level": 2}, "dr": {"enabled": False}})
    env = BittleCpuEnv(cfg, terrain_override=field, spawn_jitter_m=0.0, envelope_tier="tested")
    seen = []
    orig_step = env.step

    def step(*a, **kw):
        out = orig_step(*a, **kw)
        seen.append(out[3].terrain_h)
        return out

    env.step = step
    box_hits = []
    m = env.model
    box_geom = env.box_ids[0]
    feet = {m.geom(f"{leg}_foot").id for leg in ("rf", "lf", "rr", "lr")}
    orig_step2 = env.step

    def step2(*a, **kw):
        out = orig_step2(*a, **kw)
        box_hits.extend(1 for c in env.data.contact[: env.data.ncon]
                        if box_geom in (c.geom1, c.geom2) and (c.geom1 in feet or c.geom2 in feet))
        return out

    env.step = step2
    st = run_episode(env, GaitController.from_opencat("trF"), seed=0, seconds=6.0, command=np.array([0.12, 0, 0]))
    # terrain_h is the SUPPORT height (mean under the four feet): a 60 mm slab under a 105 mm stance holds at
    # most one pair of feet at a time, so the reference rises by exactly half the slab, never the whole 10 mm
    assert max(seen) == pytest.approx(0.005), "the feet never stood on the box"
    assert max(seen) < 0.010
    # the physics must FEEL the edge: a runtime-resized box with a stale compiled bounding radius is
    # invisible to the broadphase (zero contacts) even though the analytic height sees it
    assert len(box_hits) > 0, "no foot-box contact: the box is not colliding"


def test_terrain_metrics_are_finite_and_in_range():
    from trainer.config import apply_patch

    env = BittleCpuEnv(apply_patch(None, {"terrain": {"level": 3}}), seed=5)
    env.reset()
    for _ in range(50):
        _, _, done, info = env.step(np.random.default_rng(1).uniform(-1, 1, 8))
        assert np.isfinite(info.peak_qvel).all() and (info.peak_qvel >= 0).all()
        assert info.stalled.shape == (8,) and info.stalled.dtype == bool
        assert np.isfinite(info.foot_clearance).all() and info.terrain_h >= 0
        if done:
            break


def test_stuck_diagnostics_see_a_robot_that_never_moves_and_a_rollout_carries_limb_contacts():
    """A commanded robot holding STAND for 3 s is stuck from t=0; the rollout frames carry limb contacts."""
    from trainer.config import apply_patch
    from trainer.eval.evaluator import STUCK_MIN_S, aggregate, run_episode
    from trainer.eval.rollout import RolloutRecorder

    class Hold:
        def reset(self, env):
            pass

        def act(self, obs, env, t):
            return "target", np.asarray(spec.STAND_DEG, dtype=np.float64)

    env = BittleCpuEnv(apply_patch(None, {"dr": {"enabled": False}}), seed=0)
    rec = RolloutRecorder(fps=50, source={})
    st = run_episode(env, Hold(), seed=0, seconds=3.0, command=np.array([0.12, 0.0, 0.0]), recorder=rec)
    assert st.stuck_seconds >= 3.0 - STUCK_MIN_S - 0.1 and st.stuck_x is not None and abs(st.stuck_x) < 0.05
    agg = aggregate([st], 3.0)
    assert agg["stuck_episode_rate"] == 1.0 and agg["stuck_seconds_p50"] == st.stuck_seconds
    assert agg["stuck_limb_share"] is not None and 0.0 <= agg["stuck_limb_share"] <= 1.0
    assert rec.frames and len(rec.frames[0]["limb_contacts"]) == 8
    # a zero command is never "stuck"
    st0 = run_episode(env, Hold(), seed=0, seconds=1.0, command=np.array([0.0, 0.0, 0.0]))
    assert st0.stuck_seconds == 0.0 and st0.stuck_x is None


# ── stairs (2026-09-10) ──────────────────────────────────────────────────────

def _stairs_env(profile="up_down", spawn_jitter=0.0, seed=0, **terrain):
    from trainer.config import apply_patch

    cfg = apply_patch(None, {"terrain": {"level": 5, "stair_steps": [3, 3], "stair_rise_m": [0.018, 0.018],
                                         "stair_tread_m": [0.08, 0.08], "stair_profile": profile, **terrain},
                             "dr": {"enabled": False}})
    return BittleCpuEnv(cfg, spawn_jitter_m=spawn_jitter, envelope_tier="tested", seed=seed)


def test_stairs_down_profile_spawns_standing_on_the_platform():
    """The 'down' profile spawns the robot ON a 54 mm platform: it is lifted by the platform height, its feet
    touch the platform box (not the floor), and holding STAND for 2 s it neither falls nor sinks."""
    env = _stairs_env(profile="down")
    env.reset(command=np.zeros(3))
    m, d = env.model, env.data
    plat = tr_box_ids(env)[tr_landing_slot()]
    assert d.qpos[2] > 0.047 + 0.054 - 0.005, d.qpos[2]   # standing height ON the platform
    feet = {m.geom(f"{leg}_foot").id for leg in ("rf", "lf", "rr", "lr")}
    z0 = float(d.qpos[2])
    plat_hits, floor_hits = 0, 0
    floor = m.geom("floor").id
    for _ in range(100):
        _, _, done, info = env.step(np.zeros(8))
        assert not done
        for c in d.contact[: d.ncon]:
            pair = (c.geom1, c.geom2)
            if plat in pair and (c.geom1 in feet or c.geom2 in feet):
                plat_hits += 1
            if floor in pair and (c.geom1 in feet or c.geom2 in feet):
                floor_hits += 1
    assert plat_hits > 200 and floor_hits == 0, (plat_hits, floor_hits)
    assert abs(float(d.qpos[2]) - z0) < 0.01                    # did not sink through or hop off
    assert info.terrain_h == pytest.approx(0.054, abs=1e-6)     # support height reads the platform


def test_stairs_up_the_trot_meets_the_first_riser_and_the_feet_collide_with_it():
    """The firmware trot walked into a 3 x 18 mm flight: the FEET strike the step boxes (a contact count, not a
    height query), the support height rises off zero, and the episode neither falls nor terminates spuriously
    from the terrain-relative height check while the torso crosses the first edge."""
    from trainer.eval.evaluator import GaitController, run_episode

    env = _stairs_env()
    m = env.model
    steps = set(tr_box_ids(env)[:3])
    feet = {m.geom(f"{leg}_foot").id for leg in ("rf", "lf", "rr", "lr")}
    hits, support = [], []
    orig = env.step

    def step(*a, **kw):
        out = orig(*a, **kw)
        hits.extend(1 for c in env.data.contact[: env.data.ncon]
                    if (c.geom1 in steps or c.geom2 in steps) and (c.geom1 in feet or c.geom2 in feet))
        support.append(out[3].terrain_h)
        return out

    env.step = step
    st = run_episode(env, GaitController.from_opencat("trF"), seed=0, seconds=8.0, command=np.array([0.12, 0, 0]))
    assert len(hits) > 0, "no foot-step contact: the stair boxes are not colliding"
    # the open-loop trot swings 0.7 mm: it is STOPPED by an 18 mm riser (never reaches the landing), which is
    # exactly the degenerate baseline the stairs_v1 ratio gates are disabled for
    assert max(support) < 0.054 and st.distance_x < 0.40, (max(support), st.distance_x)
    assert st.climb_max_m == pytest.approx(max(support)) and st.descent_m >= 0.0
    assert st.seconds > 2.0 and not st.fell                     # no spurious fall from the height check


def test_stairs_spawn_jitter_never_moves_the_robot_onto_a_different_level():
    """Jitter along the runway is allowed; a jitter that would put the spawn point on step 1 is refused."""
    env = _stairs_env(spawn_jitter=0.2, seed=5)
    hs = set()
    for _ in range(30):
        env.reset(command=np.zeros(3))
        hs.add(round(env._terrain_h(env.data.qpos[0], env.data.qpos[1]), 6))
        assert env.data.qpos[2] < 0.06                          # never lifted: it always spawns on the floor level
    assert hs == {0.0}


def tr_box_ids(env):
    return env.box_ids


def tr_landing_slot():
    from trainer.env import terrain as tr

    return tr.STAIR_LANDING_SLOT


def test_stance_timeout_scores_a_real_gait_above_a_dragging_one():
    """The term's whole claim is a gradient pointing from dragging towards lifting, so check it on two
    REPLAYED behaviours rather than hand-built numbers: the firmware trot (which does pick its feet up,
    however little) against holding STAND while commanded to walk. The stander must saturate the penalty
    and the trot must collect far less; at the intended weight the trot's weighted reward is strictly
    higher, and neither is driven to the clip floor (where no gradient would survive)."""
    from trainer.config import apply_patch
    from trainer.env import spec
    from trainer.eval.evaluator import GaitController, StandController

    cfg = apply_patch(None, {"dr": {"enabled": False}})
    cmd = np.array([0.12, 0.0, 0.0])
    weights = {"tracking_lin_vel": 1.5, "stance_timeout": -0.5}

    def replay(controller):
        env = BittleCpuEnv(cfg, envelope_tier="tested", seed=0)
        obs = env.reset(command=cmd)
        controller.reset(env)
        term = rew = 0.0
        steps = 250  # 5 s
        for t in range(steps):
            kind, val = controller.act(obs, env, t)
            obs, _, done, info = (env.step(np.zeros(8), target_deg=val) if kind == "target" else env.step(val))[:4]
            term += info.terms["stance_timeout"]
            rew += float(spec.weighted_reward(np, info.terms, weights, env.cfg.control_dt))
            if done:
                break
        return term / steps, rew / steps

    trot_term, trot_rew = replay(GaitController.from_opencat("trF"))
    stand_term, stand_rew = replay(StandController())
    assert stand_term > 3 * trot_term, (trot_term, stand_term)
    assert stand_term > 0.5 * 4 * spec.STANCE_OVERDUE_CAP_S, stand_term   # the stander saturates the cap
    assert trot_rew > stand_rew, (trot_rew, stand_rew)                     # the gradient points at walking
    assert stand_rew > 0.0, "the weighted reward hit the clip floor: no gradient would survive there"
