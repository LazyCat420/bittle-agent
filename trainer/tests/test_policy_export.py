import numpy as np
import pytest

from trainer.policy.mlp import NumpyPolicy, save_policy


def test_numpy_policy_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    layers = [(rng.normal(size=(41, 32)), rng.normal(size=32)), (rng.normal(size=(32, 16)), rng.normal(size=16))]
    save_policy(tmp_path / "p.npz", layers, np.zeros(41), np.ones(41), {"action_size": 8, "obs_layout": [["x", 41]]})
    pol = NumpyPolicy.load(tmp_path / "p.npz")
    a = pol(rng.normal(size=41))
    assert a.shape == (8,) and (np.abs(a) <= 1).all()


@pytest.mark.gpu
def test_export_matches_brax_inference(tmp_path):
    """Exported numpy MLP must equal brax's deterministic policy on random obs."""
    import jax
    import jax.numpy as jp

    from trainer.train import compat  # noqa: F401
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from trainer.policy.export import export_policy

    obs_size = {"state": (41,), "privileged_state": (72,)}
    nets = ppo_networks.make_ppo_networks(obs_size, 8, preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(64, 64),
                                          policy_obs_key="state", value_obs_key="privileged_state")
    key = jax.random.PRNGKey(0)
    policy_params = nets.policy_network.init(key)
    norm = running_statistics.init_state({"state": jp.zeros(41), "privileged_state": jp.zeros(72)})
    params = (norm, policy_params, None)
    make_policy = ppo_networks.make_inference_fn(nets)
    pol = make_policy((norm, policy_params), deterministic=True)
    export_policy(str(tmp_path / "p.npz"), params, obs_layout=[("all", 41)], action_size=8, config_hash="x")
    npol = NumpyPolicy.load(tmp_path / "p.npz")
    for i in range(5):
        obs = np.random.default_rng(i).normal(size=41).astype(np.float32)
        ref, _ = pol({"state": jp.asarray(obs), "privileged_state": jp.zeros(72)}, jax.random.PRNGKey(i))
        assert np.allclose(np.asarray(ref), npol(obs), atol=1e-4)
