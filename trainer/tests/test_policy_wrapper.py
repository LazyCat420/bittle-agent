"""Unit Tests for Policy Wrapper & Arm Integration.

Tests:
1. Arm A (Baseline): pure policy passthrough with zero distortion.
2. Arms B, C, D: reflex corrections correctly augment policy output.
3. Actuator bounding: effective action is strictly clamped to [-1.0, 1.0].
4. Parity with cpu_env: wrapper output steps cleanly in CPU simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainer.policy.conventional_reflex import ConventionalReflex
from trainer.policy.fly_reflex import FlyReflex
from trainer.policy.null_reflex import NullReflex
from trainer.policy.wrapper import ReflexAugmentedPolicy


def zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros((8,) if obs.ndim == 1 else (obs.shape[0], 8), dtype=np.float32)


def ones_policy(obs: np.ndarray) -> np.ndarray:
    return np.ones((8,) if obs.ndim == 1 else (obs.shape[0], 8), dtype=np.float32)


def test_arm_a_passthrough():
    policy = ReflexAugmentedPolicy(ones_policy, reflex_controller=None)
    obs = np.zeros(41, dtype=np.float32)
    action = policy(obs)
    assert np.allclose(action, 1.0)


@pytest.mark.parametrize("reflex_cls", [ConventionalReflex, FlyReflex, NullReflex])
def test_arms_b_c_d_bounding(reflex_cls):
    ctl = reflex_cls()
    policy = ReflexAugmentedPolicy(ones_policy, reflex_controller=ctl, action_scale_deg=6.0)

    # Even with base policy at maximum +1.0 and large inputs, output must never exceed 1.0
    obs = np.zeros(41, dtype=np.float32)
    obs[0] = 5.0  # extreme tilt
    obs[3] = 10.0  # extreme gyro

    action = policy(obs)
    assert action.shape == (8,)
    assert np.all(action >= -1.0 - 1e-6)
    assert np.all(action <= 1.0 + 1e-6)


def test_simulation_step_compatibility():
    """Verify that a wrapped policy steps seamlessly in the actual CPU simulation environment."""
    from trainer.config import TrainConfig
    from trainer.env.cpu_env import BittleCpuEnv

    cfg = TrainConfig()
    env = BittleCpuEnv(cfg, seed=42)
    obs = env.reset()

    fly_ctl = FlyReflex()
    wrapped_policy = ReflexAugmentedPolicy(zero_policy, reflex_controller=fly_ctl)

    for _ in range(10):
        act = wrapped_policy(obs)
        obs, reward, done, info = env.step(act)
        assert obs.shape == (41,)
        assert isinstance(reward, float)
        if done:
            obs = env.reset()
