"""GPU (Warp) terrain tests — TRAINER_GPU_TESTS=1. These are the physics-is-sane gates."""

import io
import contextlib

import numpy as np
import pytest

pytest.importorskip("mujoco")
jax = pytest.importorskip("jax")

from trainer.config import apply_patch  # noqa: E402
from trainer.env import terrain as tr  # noqa: E402

pytestmark = pytest.mark.gpu


def _wrapped(cfg, num_envs):
    import jax.numpy as jp
    from mujoco_playground import wrapper

    from trainer.env.gpu_env import BittleGpuEnv, make_domain_randomizer
    from trainer.train import compat  # noqa: F401

    import functools

    env = BittleGpuEnv(cfg, num_envs=num_envs)
    rand = make_domain_randomizer(cfg, env.mj_model)
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    # exactly what brax PPO does: partial the per-env keys in, then wrap (vmap + episode + autoreset)
    wenv = wrapper.wrap_for_brax_training(env, randomization_fn=functools.partial(rand, rng=keys))
    return env, wenv, keys, jp


def test_per_world_gravity_is_batched_under_warp():
    """ASSUMPTION-5 in the plan: opt.gravity is a per-world field the Warp bridge honours."""
    cfg = apply_patch(None, {"terrain": {"level": 1, "slope_deg": [5.0, 15.0]}, "dr": {"enabled": False}})
    env, wenv, keys, jp = _wrapped(cfg, 8)
    st = jax.jit(wenv.reset)(keys)
    st = jax.jit(wenv.step)(st, jp.zeros((8, 8)))
    assert bool(jp.isfinite(st.obs["state"]).all())
    # the per-env model the randomiser builds: distinct gravity vectors of norm 9.81
    from trainer.env.gpu_env import make_domain_randomizer

    model, _ = make_domain_randomizer(cfg, env.mj_model)(env.mjx_model, keys)
    g = np.asarray(model.opt.gravity)
    assert g.shape == (8, 3) and np.allclose(np.linalg.norm(g, axis=1), 9.81, atol=1e-3)
    slopes = np.degrees([tr.slope_from_gravity(np, gi)[0] for gi in g])
    assert len(np.unique(np.round(slopes, 3))) == 8 and slopes.min() >= 5.0 - 1e-6 and slopes.max() <= 15.0 + 1e-6
    # the privileged terrain block is computed INSIDE the vmapped step from the per-env model:
    # it must carry each env's own gravity direction (the end-to-end proof the tilt reached physics)
    priv = np.asarray(st.obs["privileged_state"])
    assert priv.shape[1] == 41 + 3 + 3 + 8 + 8 + 4 + 4 + 1 + tr.PRIVILEGED_TERRAIN_DIM
    assert np.allclose(priv[:, -3:] * tr.G, g, atol=1e-3)


def test_per_world_boxes_are_batched_under_warp():
    cfg = apply_patch(None, {"terrain": {"level": 2}, "dr": {"enabled": False}})
    env, wenv, keys, jp = _wrapped(cfg, 8)
    from trainer.env.gpu_env import make_domain_randomizer

    model, _ = make_domain_randomizer(cfg, env.mj_model)(env.mjx_model, keys)
    ids = tr.box_body_ids(env.mj_model)
    pos = np.asarray(model.body_pos)[:, ids]
    assert pos.shape == (8, tr.MAX_BOXES, 3)
    assert (pos[:, :24, 2] > -0.1).all() and (pos[:, 24:, 2] < -0.5).all()
    assert len({tuple(np.round(pos[i, 0], 4)) for i in range(8)}) == 8  # distinct per env
    st = jax.jit(wenv.reset)(keys)
    st = jax.jit(wenv.step)(st, jp.zeros((8, 8)))
    assert bool(jp.isfinite(st.data.qpos).all())


def test_gpu_and_cpu_terrain_height_agree():
    import jax.numpy as jp

    f = tr.sample_field_numpy(np.random.default_rng(4), apply_patch(None, {"terrain": {"level": 3}}).terrain)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = rng.uniform(0, 2.5), rng.uniform(-0.3, 0.3)
        a = tr.terrain_height(np, x, y, f.box_pos, f.box_half, f.box_yaw)
        b = tr.terrain_height(jp, jp.float32(x), jp.float32(y), jp.asarray(f.box_pos), jp.asarray(f.box_half), jp.asarray(f.box_yaw))
        assert abs(float(a) - float(b)) < 1e-6


@pytest.mark.slow
def test_rough_slope_terrain_no_nan_512_envs_1000_steps(capfd):
    """512 envs x 1000 random-action steps on rocks + slope: finite state, no contact overflow."""
    cfg = apply_patch(None, {"terrain": {"level": 3}, "ppo": {"num_envs": 512}})
    env, wenv, keys, jp = _wrapped(cfg, 512)
    reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
    st = reset(keys)
    k = jax.random.PRNGKey(1)
    for _ in range(1000):
        k, ka = jax.random.split(k)
        st = step(st, jax.random.uniform(ka, (512, 8), minval=-1, maxval=1))
    assert bool(jp.isfinite(st.data.qpos).all()) and bool(jp.isfinite(st.data.qvel).all())
    assert bool((jp.abs(st.data.qvel) < 1e3).all())
    assert bool(jp.isfinite(st.obs["state"]).all()) and bool(jp.isfinite(st.obs["privileged_state"]).all())
    assert bool(jp.isfinite(st.reward).all())
    out = capfd.readouterr()
    assert "overflow" not in (out.out + out.err).lower()


def test_house_level_mixes_terrains_across_the_batch():
    """terrain.level 4 under the vmapped randomizer: some worlds flat, some tilted, some with live boxes,
    some parked — all four combinations in one batch of 64, and downhill (+x gravity) is among them."""
    import jax.numpy as jp

    from trainer.env.gpu_env import BittleGpuEnv, make_domain_randomizer

    cfg = apply_patch(None, {"terrain": {"level": 4}, "dr": {"enabled": False}})
    env = BittleGpuEnv(cfg, num_envs=64)
    rand = make_domain_randomizer(cfg, env.mj_model)
    keys = jax.random.split(jax.random.PRNGKey(1), 64)
    model, _ = rand(env.mjx_model, keys)
    g = np.asarray(model.opt.gravity)
    sloped = np.abs(g[:, 0]) + np.abs(g[:, 1]) > 1e-6
    bids = tr.box_body_ids(env.mj_model)
    live = (np.asarray(model.body_pos)[:, bids, 2] > -0.5).sum(axis=1)
    assert set(live.tolist()) <= {0, cfg.terrain.n_boxes}
    combos = {(bool(s), bool(l > 0)) for s, l in zip(sloped, live)}
    assert combos == {(False, False), (True, False), (False, True), (True, True)}
    assert (g[:, 0] > 0.5).any(), "no downhill world in the batch"
    assert 0.3 < sloped.mean() < 0.9 and 0.3 < (live > 0).mean() < 0.9


@pytest.mark.gpu
def test_pinned_protocol_field_matches_the_cpu_terrain_height():
    """dual-sim on rough: the GPU env pinned to the rough_v1 field reads the CPU field's height."""
    from trainer.config import apply_patch
    from trainer.env import terrain as tr
    from trainer.env.gpu_env import BittleGpuEnv
    from trainer.eval.gates import load_suite

    proto = load_suite("rough_v1")["protocol"]["terrain"]
    field = tr.field_from_protocol(proto)
    cfg = apply_patch(None, {"terrain": {"level": 2}, "dr": {"enabled": False}})
    env = BittleGpuEnv(cfg, num_envs=1)
    assert float(env._terrain_h(field.box_pos[0][0], field.box_pos[0][1])) == 0.0  # parked before the pin
    env.pin_terrain(field)
    for i in range(int(proto["n_boxes"])):
        x, y = field.box_pos[i][:2]
        assert abs(float(env._terrain_h(x, y)) - float(tr.terrain_height(np, x, y, field.box_pos, field.box_half, field.box_yaw))) < 1e-6
    assert abs(float(env._terrain_h(field.box_pos[0][0], field.box_pos[0][1])) - 0.012) < 1e-6


def _foot_box_contacts(env, st) -> int:
    """Foot-box contact count read from the warp contact table of a (possibly batched) state."""
    impl = getattr(st.data, "_impl", st.data)
    geom = np.asarray(impl.contact__geom).reshape(-1, 2)
    n = int(np.asarray(impl.nacon).reshape(-1).sum())
    box = set(int(i) for i in np.asarray(env._box_gids))
    feet = set(int(i) for i in np.asarray(env._foot_gids))
    return sum(1 for g1, g2 in geom[:n] if ({int(g1), int(g2)} & box) and ({int(g1), int(g2)} & feet))


@pytest.mark.gpu
def test_the_feet_actually_collide_with_the_boxes_under_warp():
    """The engine, not the analytic height, must feel the rocks: a robot standing on the pinned rough_v1
    field with a box under a foot reports a foot-box contact and the box geom sits where the field says
    (before 2026-09-10 the static box geoms stayed parked at z=-1 in data: 0 contacts, feet through rocks)."""
    from trainer.config import apply_patch
    from trainer.env import terrain as tr
    from trainer.env.gpu_env import BittleGpuEnv
    from trainer.eval.gates import load_suite
    import jax.numpy as jp

    field = tr.field_from_protocol(load_suite("rough_v1")["protocol"]["terrain"])
    # a wide 12 mm slab right under the spawn so the feet must land on it
    field.box_pos[0] = [0.0, 0.0, tr.PLANE_Z + 0.012 - tr.BOX_HALF_Z]
    field.box_half[0] = [0.12, 0.12, tr.BOX_HALF_Z]
    field.box_yaw[0] = 0.0
    cfg = apply_patch(None, {"terrain": {"level": 2, "spawn_jitter_m": 0.0}, "dr": {"enabled": False}})
    env = BittleGpuEnv(cfg, num_envs=1)
    env.pin_terrain(field)
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    gx = np.asarray(st.data.geom_xpos)[np.asarray(env._box_gids)[0]]
    assert np.allclose(gx, field.box_pos[0], atol=1e-5), f"box geom parked at {gx}"
    step = jax.jit(env.step)
    hits = 0
    for _ in range(25):  # half a second: the robot settles onto the slab
        st = step(st, jp.zeros(8))
        hits += _foot_box_contacts(env, st)
    assert hits > 0, "no foot-box contact under warp: the rocks are phantoms"
    torso_z = float(st.data.qpos[2])
    assert torso_z > 0.012 + 0.03, f"torso at {torso_z:.3f} m: standing on the floor, not on the 12 mm slab"


@pytest.mark.gpu
def test_training_path_boxes_collide_per_env():
    """The brax training path (wrapper + per-env randomizer): every env's own field is what the feet hit."""
    cfg = apply_patch(None, {"terrain": {"level": 2}, "dr": {"enabled": False}})
    env, wenv, keys, jp_ = _wrapped(cfg, 4)
    st = jax.jit(wenv.reset)(keys)
    ids = tr.box_geom_ids(env.mj_model)
    gx = np.asarray(st.data.geom_xpos)[:, ids[:24], 2]
    assert (gx > -0.1).all(), "box geoms parked in the batched data"
    assert len({tuple(np.round(np.asarray(st.data.geom_xpos)[i, ids[0]], 4)) for i in range(4)}) == 4
