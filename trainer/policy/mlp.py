"""Numpy inference for an exported policy (``policy.npz``).

The file carries everything a robot-side runner needs: layer weights, the
observation normaliser, the obs layout, the action post-processing and the
config hash the policy was trained under.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], obs_mean: np.ndarray, obs_std: np.ndarray,
                 meta: dict[str, Any]):
        self.layers = layers
        self.obs_mean = obs_mean.astype(np.float64)
        self.obs_std = obs_std.astype(np.float64)
        self.meta = meta
        self.action_size = int(meta.get("action_size", 8))

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        layers = []
        for i in range(int(meta["n_layers"])):
            layers.append((z[f"w{i}"], z[f"b{i}"]))
        return cls(layers, z["obs_mean"], z["obs_std"], meta)

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std
        n = len(self.layers)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < n - 1:
                x = silu(x)
        loc = x[: self.action_size]
        return np.tanh(loc)  # tanh_normal, deterministic mode


def save_policy(path: str | Path, layers: list[tuple[np.ndarray, np.ndarray]], obs_mean: np.ndarray,
                obs_std: np.ndarray, meta: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {"obs_mean": np.asarray(obs_mean), "obs_std": np.asarray(obs_std)}
    for i, (w, b) in enumerate(layers):
        arrays[f"w{i}"] = np.asarray(w)
        arrays[f"b{i}"] = np.asarray(b)
    meta = dict(meta, n_layers=len(layers))
    arrays["meta_json"] = np.array(json.dumps(meta))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
