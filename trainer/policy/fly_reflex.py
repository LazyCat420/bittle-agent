"""Arm C: Engineered Fly-Derived Reflex Controller for Bittle.

Based on:
1. Drosophila Haltere Circuit: ultra-fast angular acceleration detection
   (d(omega)/dt) for sudden slip and perturbation detection.
2. Drosophila Central Complex (EB/PB) Dual-Ring Attractor Network:
   Two coupled 16-neuron continuous attractor rings (Roll Ring & Pitch Ring)
   performing angular rate integration and non-linear bump stabilization
   (c.f. Green et al., Nature 2017).

Inputs:
- obs: 41-dim observation array (or batched (B, 41))
  * obs[..., 0:3]: gravity projection vector (gx, gy, gz)
  * obs[..., 3:6]: scaled angular velocity (wx, wy, wz) * 0.25

Outputs:
- delta_theta: (8,) or (B, 8) joint angle trims in bittle-agent degrees,
  bounded strictly within [-6.0, +6.0] deg/step.
"""

from __future__ import annotations

import math
import numpy as np


class FlyReflex:
    """Engineered fly-derived dual-ring attractor & haltere reflex stabilizer."""

    def __init__(
        self,
        n_neurons_per_ring: int = 16,
        dt: float = 0.02,  # 50 Hz control step
        j_exc: float = 2.0,
        j_inh: float = -1.5,
        haltere_jerk_thresh: float = 15.0,  # rad/s^2 slip threshold
        max_trim_deg: float = 6.0,
    ) -> None:
        self.n = int(n_neurons_per_ring)
        self.dt = float(dt)
        self.haltere_jerk_thresh = float(haltere_jerk_thresh)
        self.max_trim_deg = float(max_trim_deg)

        # Precompute circular phase angles and cosine recurrent connectivity
        self.angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        diffs = self.angles[:, np.newaxis] - self.angles[np.newaxis, :]
        # W_ij = J_inh + J_exc * cos(phi_i - phi_j)
        self.w_rec = (j_inh + j_exc * np.cos(diffs)).astype(np.float32)

        # State storage (for single or batched instances)
        # Membrane activations u_pitch, u_roll: shape (B, N)
        self.u_pitch = np.zeros((1, self.n), dtype=np.float32)
        self.u_roll = np.zeros((1, self.n), dtype=np.float32)
        self.prev_omega = np.zeros((1, 3), dtype=np.float32)
        self.initialized = False

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal attractor states and sensory buffers."""
        self.u_pitch = np.zeros((batch_size, self.n), dtype=np.float32)
        self.u_roll = np.zeros((batch_size, self.n), dtype=np.float32)
        self.prev_omega = np.zeros((batch_size, 3), dtype=np.float32)
        self.initialized = True

    def compute_trim(self, obs: np.ndarray) -> np.ndarray:
        """Step the ring attractors and compute bounded joint angle trims."""
        arr = np.asarray(obs, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]

        batch_size = arr.shape[0]
        if not self.initialized or self.u_pitch.shape[0] != batch_size:
            self.reset(batch_size)

        # Extract IMU inputs
        gx = arr[:, 0]
        gy = arr[:, 1]
        omega = arr[:, 3:6] / 0.25  # unscale to physical rad/s
        wx = omega[:, 0]
        wy = omega[:, 1]

        # 1. Haltere Gyroscopic Circuit: compute angular acceleration d(omega)/dt
        d_omega_dt = (omega - self.prev_omega) / self.dt
        self.prev_omega = omega.copy()
        jerk_magnitude = np.linalg.norm(d_omega_dt, axis=-1)  # (B,)

        # Emergency crouch gain if sudden slip acceleration detected
        crouch_boost = np.where(jerk_magnitude > self.haltere_jerk_thresh, 1.5, 1.0)

        # 2. Dual-Ring Attractor Dynamics
        # Sensory input projections: map tilt angle into sensory driving vector around ring
        # Target angle phi = arctan2(gy, gz) etc., or linear tilt angle approximation
        tilt_pitch = np.clip(gx, -1.0, 1.0) * (np.pi / 2.0)
        tilt_roll = np.clip(gy, -1.0, 1.0) * (np.pi / 2.0)

        s_pitch = np.cos(self.angles[np.newaxis, :] - tilt_pitch[:, np.newaxis])
        s_roll = np.cos(self.angles[np.newaxis, :] - tilt_roll[:, np.newaxis])

        # Shift ring bump by velocity integration (P-EN heading shift mechanism)
        tau = 0.05
        decay = math.exp(-self.dt / tau)

        # Recurrent updates
        act_pitch = np.maximum(0.0, self.u_pitch)
        act_roll = np.maximum(0.0, self.u_roll)

        rec_pitch = act_pitch @ self.w_rec.T
        rec_roll = act_roll @ self.w_rec.T

        self.u_pitch = self.u_pitch * decay + (rec_pitch + 2.0 * s_pitch) * (1.0 - decay)
        self.u_roll = self.u_roll * decay + (rec_roll + 2.0 * s_roll) * (1.0 - decay)

        # 3. Readout Decoder: compute population vector / center-of-mass phase and amplitude
        sin_sum_p = np.sum(np.maximum(0.0, self.u_pitch) * np.sin(self.angles), axis=-1)
        cos_sum_p = np.sum(np.maximum(0.0, self.u_pitch) * np.cos(self.angles), axis=-1)
        bump_phase_p = np.arctan2(sin_sum_p, cos_sum_p)

        sin_sum_r = np.sum(np.maximum(0.0, self.u_roll) * np.sin(self.angles), axis=-1)
        cos_sum_r = np.sum(np.maximum(0.0, self.u_roll) * np.cos(self.angles), axis=-1)
        bump_phase_r = np.arctan2(sin_sum_r, cos_sum_r)

        # Readout mapping: bump phase encodes tilt error, damped by angular velocity
        pitch_trim = -(8.0 * bump_phase_p + 2.0 * wy) * crouch_boost
        roll_trim = -(8.0 * bump_phase_r + 2.0 * wx) * crouch_boost

        # Policy joint order: (lf_sh, rf_sh, rr_sh, lr_sh, lf_kn, rf_kn, rr_kn, lr_kn)
        delta_theta = np.zeros((batch_size, 8), dtype=np.float32)

        # Front Shoulders: lf=0, rf=1
        delta_theta[:, 0] += pitch_trim + roll_trim
        delta_theta[:, 1] += pitch_trim - roll_trim

        # Rear Shoulders: rr=2, lr=3
        delta_theta[:, 2] += -pitch_trim - roll_trim
        delta_theta[:, 3] += -pitch_trim + roll_trim

        # Knees: lf=4, rf=5, rr=6, lr=7
        delta_theta[:, 4] += pitch_trim + roll_trim
        delta_theta[:, 5] += pitch_trim - roll_trim
        delta_theta[:, 6] += -pitch_trim - roll_trim
        delta_theta[:, 7] += -pitch_trim + roll_trim

        # Strict hardware clipping
        clipped = np.clip(delta_theta, -self.max_trim_deg, self.max_trim_deg)
        return clipped[0] if is_1d else clipped
