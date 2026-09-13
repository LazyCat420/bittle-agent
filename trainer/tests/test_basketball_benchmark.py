"""Unit test for basketball balance benchmark evaluation.

Verifies:
1. ball_v1 suite loads cleanly and protocol maps to bittle_basketball.xml.
2. An episode can be simulated with FlyReflex on the basketball model.
3. Output metrics include fall_rate, centre_drift_m, mean_tilt_deg, rms_vz.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainer.config import TrainConfig
from trainer.eval.evaluator import PolicyController, evaluate_protocol, protocol_kwargs
from trainer.eval.gates import load_suite, suite_protocol
from trainer.policy.fly_reflex import FlyReflex
from trainer.policy.wrapper import ReflexAugmentedPolicy


def zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros((8,) if obs.ndim == 1 else (obs.shape[0], 8), dtype=np.float32)


def test_basketball_suite_evaluation():
    suite = load_suite("ball_v1")
    proto = suite_protocol(suite)
    assert proto["model"] == "bittle_basketball.xml"

    cfg = TrainConfig(task="ball_balance")
    fly_ctl = FlyReflex(max_trim_deg=4.0)
    policy = ReflexAugmentedPolicy(zero_policy, reflex_controller=fly_ctl)
    ctrl = PolicyController(policy)

    common = protocol_kwargs(proto)
    assert common["variant"] == "basketball"

    # Evaluate 2 short episodes (1.0 s) to verify end-to-end evaluation pipeline with rollout recording
    stats, rollouts = evaluate_protocol(
        cfg, ctrl, n_episodes=2, seed_start=0, seconds=1.0,
        command=[0.0, 0.0, 0.0], record_seeds={0}, **common
    )

    assert len(stats) == 2
    for ep in stats:
        assert isinstance(ep.fell, bool)
        assert ep.seconds <= 1.05
        assert isinstance(ep.centre_drift_m, float)
        assert isinstance(ep.tilt_deg, float)
        assert not np.isnan(ep.tilt_deg)

    assert 0 in rollouts, "Rollout seed 0 was not recorded"
    ro = rollouts[0]
    assert len(ro["frames"]) > 0, "No frames recorded in rollout"
    first_frame = ro["frames"][0]
    assert "ball_pos_m" in first_frame, "ball_pos_m missing from rollout frame"
    assert len(first_frame["ball_pos_m"]) == 3, f"Invalid ball_pos_m: {first_frame['ball_pos_m']}"
    assert "ball_rpy_deg" in first_frame, "ball_rpy_deg missing from rollout frame"
