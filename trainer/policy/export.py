"""Export brax PPO params -> policy.npz (numpy MLP + normaliser + layout)."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from .mlp import save_policy


def _flatten_mlp(params: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    p = params["params"] if "params" in params else params
    keys = sorted((k for k in p if k.startswith("hidden_")), key=lambda k: int(k.split("_")[1]))
    return [(np.asarray(p[k]["kernel"]), np.asarray(p[k]["bias"])) for k in keys]


def export_policy(path: str, params: tuple, *, obs_layout: list[tuple[str, int]], action_size: int,
                  config_hash: str, extra: dict[str, Any] | None = None) -> None:
    """``params`` = (normalizer_state, policy_params, value_params) as returned by brax PPO."""
    normalizer, policy_params = params[0], params[1]
    normalizer = jax.device_get(normalizer)
    mean = normalizer.mean["state"] if isinstance(normalizer.mean, dict) else normalizer.mean
    std = normalizer.std["state"] if isinstance(normalizer.std, dict) else normalizer.std
    layers = _flatten_mlp(jax.device_get(policy_params))
    meta = {
        "action_size": action_size,
        "obs_layout": obs_layout,
        "obs_size": int(sum(n for _, n in obs_layout)),
        "activation": "silu",
        "distribution": "tanh_normal",
        "config_hash": config_hash,
        "obs_key": "state",
    }
    if extra:
        meta.update(extra)
    save_policy(path, layers, np.asarray(mean), np.asarray(std), meta)
