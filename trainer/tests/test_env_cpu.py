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
    assert max(seen) == 0.010, "the torso never passed over the box"
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
