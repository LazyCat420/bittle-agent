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
