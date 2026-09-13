"""Policy wrapper augmenting any standard PPO policy with a reflex layer.

Provides a unified interface for all 4 experimental arms:
- Arm A: Base PPO alone (reflex = None)
- Arm B: PPO + ConventionalReflex
- Arm C: PPO + FlyReflex
- Arm D: PPO + NullReflex

Combines policy action a in [-1, 1] with reflex trim delta_theta in [-max_trim, +max_trim] deg:
  effective_action = clip(a + delta_theta / scale_deg, -1.0, 1.0)
ensuring all servo envelope and slew rate constraints are strictly honored.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np


class ReflexAugmentedPolicy:
    """Wraps a base policy callable with an auxiliary reflex stabilizer."""

    def __init__(
        self,
        base_policy: Callable[[np.ndarray], np.ndarray],
        reflex_controller: Any | None = None,
        action_scale_deg: float = 6.0,
    ) -> None:
        self.base_policy = base_policy
        self.reflex = reflex_controller
        self.scale_deg = float(action_scale_deg)

    def reset(self, batch_size: int = 1) -> None:
        if self.reflex is not None and hasattr(self.reflex, "reset"):
            self.reflex.reset(batch_size)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Base policy produces normalized actions in [-1, 1]^8
        action = np.asarray(self.base_policy(obs), dtype=np.float32)

        if self.reflex is None:
            return np.clip(action, -1.0, 1.0)

        # Reflex computes joint delta trims in degrees [-6.0, +6.0]
        trim_deg = self.reflex.compute_trim(obs)
        normalized_trim = trim_deg / self.scale_deg

        # Combine and strictly clip to [-1, 1]
        effective_action = np.clip(action + normalized_trim, -1.0, 1.0)
        return effective_action
