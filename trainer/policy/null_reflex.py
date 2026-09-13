"""Arm D: Randomized Rewired Null Network Reflex Controller for Bittle.

Acts as a scientific null control matching Arm C (FlyReflex) in:
- Neuron count: 32 recurrent units (matching the 2 x 16 ring neurons).
- Input dimensions & sensory projections.
- Activation functions: ReLU max(0, u) and leaky integration time constants (tau = 0.05).
- Output bounding: strictly [-6.0, +6.0] deg/step.

Key Difference:
The recurrent synaptic weight matrix W_null is a randomized, unstructured
Gaussian matrix (scaled to match spectral radius) rather than the structured
cosine ring-attractor topology of Arm C. This isolates whether performance
improvements stem from the biological ring topology or merely from having
a recurrent neural reservoir.
"""

from __future__ import annotations

import math
import numpy as np


class NullReflex:
    """Randomized recurrent null model matching Arm C size and dynamics."""

    def __init__(
        self,
        n_neurons: int = 32,
        dt: float = 0.02,
        seed: int = 42,
        max_trim_deg: float = 6.0,
    ) -> None:
        self.n = int(n_neurons)
        self.dt = float(dt)
        self.max_trim_deg = float(max_trim_deg)

        # Generate fixed random recurrent weights with spectral radius ~0.9
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((self.n, self.n)).astype(np.float32)
        radius = np.max(np.abs(np.linalg.eigvals(w)))
        self.w_rec = (w / max(radius, 1e-6)) * 0.9

        # Random input projections from (gx, gy, wx, wy) -> n_neurons
        self.w_in = rng.standard_normal((self.n, 4)).astype(np.float32) * 0.5

        # Random readout projection: n_neurons -> 8 joint trims
        self.w_out = rng.standard_normal((8, self.n)).astype(np.float32) * 0.5

        self.u = np.zeros((1, self.n), dtype=np.float32)
        self.initialized = False

    def reset(self, batch_size: int = 1) -> None:
        self.u = np.zeros((batch_size, self.n), dtype=np.float32)
        self.initialized = True

    def compute_trim(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]

        batch_size = arr.shape[0]
        if not self.initialized or self.u.shape[0] != batch_size:
            self.reset(batch_size)

        # Extract 4 key attitude signals: gx, gy, wx, wy
        gx = arr[:, 0:1]
        gy = arr[:, 1:2]
        wx = (arr[:, 3:4] / 0.25) * 0.1
        wy = (arr[:, 4:5] / 0.25) * 0.1
        inputs = np.concatenate([gx, gy, wx, wy], axis=-1)  # (B, 4)

        # Leaky integration
        tau = 0.05
        decay = math.exp(-self.dt / tau)

        act = np.maximum(0.0, self.u)  # (B, N)
        rec = act @ self.w_rec.T
        sensory = inputs @ self.w_in.T

        self.u = self.u * decay + (rec + sensory) * (1.0 - decay)

        # Readout to 8 joint trims
        raw_trim = np.maximum(0.0, self.u) @ self.w_out.T  # (B, 8)
        clipped = np.clip(raw_trim, -self.max_trim_deg, self.max_trim_deg)
        return clipped[0] if is_1d else clipped
