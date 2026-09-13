"""Arm B: Conventional Feedback Reflex Controller for Bittle.

Implements a classical Proportional-Derivative (PD) attitude-leveling reflex
operating on deployable onboard IMU signals (gravity projection vector g and
gyroscope angular velocity omega).

Inputs:
- obs: 41-dim observation array (or batched (B, 41)), where:
  * obs[..., 0:3]: projected gravity vector (gx, gy, gz) in body frame
  * obs[..., 3:6]: scaled angular velocity (wx, wy, wz) * 0.25 in body frame

Outputs:
- delta_theta: (8,) or (B, 8) joint angle trims in bittle-agent degrees,
  bounded strictly within [-6.0, +6.0] deg/step.
  Policy joint order: (lf_sh, rf_sh, rr_sh, lr_sh, lf_kn, rf_kn, rr_kn, lr_kn).
"""

from __future__ import annotations

import numpy as np


class ConventionalReflex:
    """PD attitude leveling controller outputting bounded joint trims."""

    def __init__(
        self,
        kp_pitch: float = 8.0,
        kd_pitch: float = 2.0,
        kp_roll: float = 8.0,
        kd_roll: float = 2.0,
        max_trim_deg: float = 6.0,
    ) -> None:
        self.kp_pitch = float(kp_pitch)
        self.kd_pitch = float(kd_pitch)
        self.kp_roll = float(kp_roll)
        self.kd_roll = float(kd_roll)
        self.max_trim_deg = float(max_trim_deg)

    def compute_trim(self, obs: np.ndarray) -> np.ndarray:
        """Compute corrective joint target trims from observation vector.

        Supports single observation (41,) or batched observations (B, 41).
        """
        arr = np.asarray(obs, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]

        # Extract IMU quantities
        # gx: forward tilt (pitch down > 0, pitch up < 0)
        # gy: lateral tilt (roll left > 0, roll right < 0)
        gx = arr[:, 0]
        gy = arr[:, 1]

        # Gyro in spec.py is scaled by 0.25; unscale to get physical rad/s units
        wx = arr[:, 3] / 0.25  # roll rate (rad/s)
        wy = arr[:, 4] / 0.25  # pitch rate (rad/s)

        # PD attitude error terms (target level is gx = 0, gy = 0)
        pitch_trim = -(self.kp_pitch * gx + self.kd_pitch * wy)
        roll_trim = -(self.kp_roll * gy + self.kd_roll * wx)

        # Policy joint order: (lf_sh, rf_sh, rr_sh, lr_sh, lf_kn, rf_kn, rr_kn, lr_kn)
        # Pitch response:
        # If pitching forward (nose down, gx > 0), pitch_trim < 0:
        # Front legs extend (push nose up), rear legs compress (lower rear)
        # Roll response:
        # If rolling left (gy > 0), roll_trim < 0:
        # Left legs extend (push left up), right legs compress
        batch_size = arr.shape[0]
        delta_theta = np.zeros((batch_size, 8), dtype=np.float32)

        # Front Shoulders: lf=0, rf=1
        delta_theta[:, 0] += pitch_trim + roll_trim   # lf_sh
        delta_theta[:, 1] += pitch_trim - roll_trim   # rf_sh

        # Rear Shoulders: rr=2, lr=3
        delta_theta[:, 2] += -pitch_trim - roll_trim  # rr_sh
        delta_theta[:, 3] += -pitch_trim + roll_trim  # lr_sh

        # Knees: lf=4, rf=5, rr=6, lr=7 (knee extension counters pitch/roll)
        delta_theta[:, 4] += pitch_trim + roll_trim   # lf_kn
        delta_theta[:, 5] += pitch_trim - roll_trim   # rf_kn
        delta_theta[:, 6] += -pitch_trim - roll_trim  # rr_kn
        delta_theta[:, 7] += -pitch_trim + roll_trim  # lr_kn

        # Strict hardware slew rate bounding
        clipped = np.clip(delta_theta, -self.max_trim_deg, self.max_trim_deg)
        return clipped[0] if is_1d else clipped
