"""Unit Tests for Stage 1 Reflex Controllers: Arms B, C, and D.

Validates:
1. Strict hardware bounding: trims never exceed [-6.0, +6.0] deg/step.
2. Batching parity: (41,) and (B, 41) produce identical results.
3. Symmetry & physical response:
   - Pitch tilt generates symmetric left-right trims.
   - Roll tilt generates antisymmetric left-right trims.
4. Fly Haltere jerk response: sudden angular acceleration amplifies reflex output.
5. Microsecond execution latency: each reflex step completes in < 0.2 ms.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from trainer.policy.conventional_reflex import ConventionalReflex
from trainer.policy.fly_reflex import FlyReflex
from trainer.policy.null_reflex import NullReflex


@pytest.fixture
def controllers():
    return {
        "arm_b": ConventionalReflex(max_trim_deg=6.0),
        "arm_c": FlyReflex(max_trim_deg=6.0),
        "arm_d": NullReflex(max_trim_deg=6.0),
    }


def _dummy_obs(gx: float = 0.0, gy: float = 0.0, wx: float = 0.0, wy: float = 0.0) -> np.ndarray:
    obs = np.zeros(41, dtype=np.float32)
    obs[0] = gx
    obs[1] = gy
    obs[2] = -1.0  # normal gravity pointing down
    obs[3] = wx * 0.25  # gyro scaled
    obs[4] = wy * 0.25
    return obs


@pytest.mark.parametrize("arm", ["arm_b", "arm_c", "arm_d"])
def test_strict_bounding_under_extreme_inputs(controllers, arm):
    ctl = controllers[arm]
    # Test extreme gravity and gyro inputs
    for scale in (0.0, 1.0, 10.0, -10.0, 100.0):
        obs = _dummy_obs(gx=scale, gy=scale, wx=scale * 5.0, wy=scale * 5.0)
        trim = ctl.compute_trim(obs)
        assert trim.shape == (8,), f"Expected (8,) output, got {trim.shape}"
        assert np.all(trim >= -6.0 - 1e-5), f"{arm} exceeded lower bound: {trim.min()}"
        assert np.all(trim <= 6.0 + 1e-5), f"{arm} exceeded upper bound: {trim.max()}"


@pytest.mark.parametrize("arm", ["arm_b", "arm_c", "arm_d"])
def test_batching_parity(controllers, arm):
    ctl = controllers[arm]
    obs1 = _dummy_obs(gx=0.3, gy=-0.2, wx=1.0, wy=-0.5)
    obs2 = _dummy_obs(gx=-0.4, gy=0.5, wx=-2.0, wy=1.5)

    if hasattr(ctl, "reset"):
        ctl.reset(1)
    trim1 = ctl.compute_trim(obs1)

    if hasattr(ctl, "reset"):
        ctl.reset(1)
    trim2 = ctl.compute_trim(obs2)

    if hasattr(ctl, "reset"):
        ctl.reset(2)
    batch_obs = np.stack([obs1, obs2], axis=0)
    batch_trim = ctl.compute_trim(batch_obs)

    assert batch_trim.shape == (2, 8)
    np.testing.assert_allclose(batch_trim[0], trim1, atol=1e-5)
    np.testing.assert_allclose(batch_trim[1], trim2, atol=1e-5)


def test_attitude_symmetry_conventional(controllers):
    ctl = controllers["arm_b"]

    # Pure forward pitch: nose down (gx > 0)
    obs_pitch = _dummy_obs(gx=0.4, gy=0.0)
    trim_p = ctl.compute_trim(obs_pitch)
    # Left and right front shoulders (idx 0, 1) should match symmetrically
    assert math.isclose(trim_p[0], trim_p[1], abs_tol=1e-5), "Front shoulders not symmetric under pitch"
    # Left and right rear shoulders (idx 2, 3) should match symmetrically
    assert math.isclose(trim_p[2], trim_p[3], abs_tol=1e-5), "Rear shoulders not symmetric under pitch"

    # Pure left roll: left side down (gy > 0)
    obs_roll = _dummy_obs(gx=0.0, gy=0.4)
    trim_r = ctl.compute_trim(obs_roll)
    # Left and right front shoulders should be opposite (antisymmetric)
    assert math.isclose(trim_r[0], -trim_r[1], abs_tol=1e-5), "Front shoulders not antisymmetric under roll"


def test_fly_haltere_jerk_boost(controllers):
    ctl = controllers["arm_c"]
    ctl.reset(1)

    # Step 1: Smooth constant gyro rate
    obs1 = _dummy_obs(gx=0.2, gy=0.0, wy=1.0)
    trim1 = ctl.compute_trim(obs1)

    # Step 2: Sudden violent angular acceleration (jerk)
    obs2 = _dummy_obs(gx=0.2, gy=0.0, wy=20.0)  # sudden jump in gyro rate -> high d_omega/dt
    trim2 = ctl.compute_trim(obs2)

    # High jerk must trigger increased corrective trim magnitude
    assert np.max(np.abs(trim2)) >= np.max(np.abs(trim1))


@pytest.mark.parametrize("arm", ["arm_b", "arm_c", "arm_d"])
def test_execution_latency(controllers, arm):
    ctl = controllers[arm]
    obs = _dummy_obs(gx=0.1, gy=-0.1, wx=0.5, wy=0.5)

    # Warmup
    for _ in range(10):
        ctl.compute_trim(obs)

    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ctl.compute_trim(obs)
    dt_per_step_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Must execute in < 0.2 ms per step on CPU
    assert dt_per_step_ms < 0.2, f"{arm} took {dt_per_step_ms:.3f} ms per step, expected < 0.2 ms"


import math
