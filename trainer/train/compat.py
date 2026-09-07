"""Compatibility shims between brax 0.14.2 and JAX 0.11.

JAX 0.11 removed ``jax.device_put_replicated`` (pmap migration); brax's PPO
still calls it to replicate the training state across local devices. This
restores it with the documented drop-in replacement. Import this module
BEFORE ``brax.training``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _device_put_replicated(x, devices):
    devices = list(devices)
    mesh = jax.sharding.Mesh(np.array(devices), ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))

    def rep(a):
        a = jnp.asarray(a)
        return jax.device_put(jnp.broadcast_to(a, (len(devices),) + a.shape), sharding)

    return jax.tree_util.tree_map(rep, x)


if not hasattr(jax, "device_put_replicated") or getattr(jax, "_bittle_shim", False):
    pass
try:
    jax.device_put_replicated  # noqa: B018 - probes the deprecation getattr
except AttributeError:
    jax.device_put_replicated = _device_put_replicated  # type: ignore[attr-defined]
    jax._bittle_shim = True  # type: ignore[attr-defined]
