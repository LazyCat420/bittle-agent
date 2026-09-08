"""Terrain — the ``spec.py`` of ground geometry. ONE source of truth for both engines.

Two mechanisms, both per-env and both expressible through fields that MuJoCo
Warp batches per world (``opt.gravity``, ``geom_pos/size/quat``):

* **Incline = a tilted world.** The contact plane stays normal to world +z and
  gravity is rotated instead. Every height / vertical-velocity / up-vector
  quantity in ``spec.py`` is then measured along the terrain normal with no
  code change, and the IMU gravity vector is bit-for-bit what a real slope
  produces. A slope run uses the same XML as a flat run.
* **Rocks / edges = yaw-only half-buried boxes** parked out of reach in the
  ``*_terrain.xml`` variants and moved into place per env. Yaw-only keeps the
  analytic ``terrain_height`` exact; half-buried means there is never a gap
  under a box.

Every function takes ``xp`` (numpy or jax.numpy) and touches no simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: The ground plane of static/assets/bittle.xml sits at z = -0.01 (foot sites rest at z = 0).
PLANE_Z = -0.01
#: Boxes baked into the terrain XML variants; a config may enable up to this many.
MAX_BOXES = 32
#: Parked (disabled) box: 1 m below the floor — cannot touch anything. Its half-extents are the
#: LARGEST a config may request (box_size_m <= 0.15): MuJoCo fixes each geom's bounding radius at
#: compile time, so a box may shrink at run time but never grow past its compiled size.
PARKED_POS = np.array([0.0, 0.0, -1.0])
PARKED_HALF = np.array([0.15, 0.15, 0.02])
#: Half height of an enabled box; the box top sits at PLANE_Z + protrusion.
BOX_HALF_Z = 0.02
G = 9.81
KINDS = ("flat", "slope", "rough", "rough_slope")
#: Body-frame (x, y) offsets of the critic's 3x3 height scan.
SCAN_OFFSETS = np.array([(dx, dy) for dx in (-0.06, 0.0, 0.06) for dy in (-0.06, 0.0, 0.06)])
#: Size of the critic-only terrain block appended to ``privileged_state``.
PRIVILEGED_TERRAIN_DIM = len(SCAN_OFFSETS) + 4 + 3


def has_boxes(kind: str) -> bool:
    return kind in ("rough", "rough_slope")


def has_slope(kind: str) -> bool:
    return kind in ("slope", "rough_slope")


def variant_for(kind: str, engine: str) -> str:
    """Model variant name for a terrain kind: flat and slope reuse the flat XML."""
    if kind not in KINDS:
        raise ValueError(f"unknown terrain kind {kind!r}; expected one of {KINDS}")
    return engine if not has_boxes(kind) else f"{engine}_terrain"


# ── gravity ────────────────────────────────────────────────────────────────

def gravity_for_slope(xp, slope_rad, yaw_rad):
    """World gravity for a robot walking UP a slope of ``slope_rad`` whose uphill
    direction is at ``yaw_rad`` from +x. Gravity pulls back along the slope."""
    s, c = xp.sin(slope_rad), xp.cos(slope_rad)
    return G * xp.stack([-s * xp.cos(yaw_rad), -s * xp.sin(yaw_rad), -c])


def slope_from_gravity(xp, gravity):
    """Inverse of ``gravity_for_slope``: (slope_rad, yaw_rad)."""
    g = gravity / xp.linalg.norm(gravity)
    slope = xp.arccos(xp.clip(-g[2], -1.0, 1.0))
    yaw = xp.arctan2(-g[1], -g[0])
    return slope, yaw


def uphill_xy(xp, gravity):
    """Unit-uphill direction scaled by sin(slope): exactly 0 on flat ground."""
    return -gravity[:2] / G


# ── boxes ──────────────────────────────────────────────────────────────────

def yaw_from_quat(xp, q):
    """Yaw about world z of a (w, x, y, z) quaternion (boxes are yaw-only)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_from_yaw(xp, yaw):
    zeros = xp.zeros_like(yaw)
    return xp.stack([xp.cos(yaw / 2), zeros, zeros, xp.sin(yaw / 2)], axis=-1)


def terrain_height(xp, x, y, box_pos, box_half, box_yaw):
    """Surface height ABOVE THE NOMINAL PLANE at (x, y): 0 on bare ground, the
    tallest box top under the point otherwise. Exact for yaw-only boxes."""
    if box_pos.shape[0] == 0:
        return xp.zeros(())
    dx = x - box_pos[:, 0]
    dy = y - box_pos[:, 1]
    c, s = xp.cos(box_yaw), xp.sin(box_yaw)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    inside = (xp.abs(u) <= box_half[:, 0]) & (xp.abs(v) <= box_half[:, 1])
    top = box_pos[:, 2] + box_half[:, 2] - PLANE_Z
    return xp.max(xp.where(inside, top, 0.0))


def height_scan(xp, x, y, yaw, box_pos, box_half, box_yaw):
    """Terrain height at the 3x3 body-frame scan points around (x, y)."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    out = []
    for dx, dy in SCAN_OFFSETS:
        px = x + c * dx - s * dy
        py = y + s * dx + c * dy
        out.append(terrain_height(xp, px, py, box_pos, box_half, box_yaw))
    return xp.stack(out)


#: foot sphere radius (build_models.FOOT_RADIUS); the sphere bottom is what touches the ground
FOOT_RADIUS = 0.006


def foot_clearance(xp, foot_geom_pos, box_pos, box_half, box_yaw):
    """Per-foot height of the SPHERE BOTTOM above the local terrain surface (0 when standing on it).

    Uses the foot geom centre, not the FK site: the site sits at the leg tip and swings up to
    10 mm relative to the rubber paw's contact point as the foot rotates, so a site-based
    clearance reads negative during a perfectly good swing.
    """
    return xp.stack([foot_geom_pos[i, 2] - FOOT_RADIUS - PLANE_Z
                     - terrain_height(xp, foot_geom_pos[i, 0], foot_geom_pos[i, 1], box_pos, box_half, box_yaw)
                     for i in range(foot_geom_pos.shape[0])])


# ── per-env field ──────────────────────────────────────────────────────────

@dataclass
class TerrainField:
    """One env's terrain in the units both engines use (world metres / radians)."""

    gravity: np.ndarray  # (3,)
    box_pos: np.ndarray  # (K, 3)
    box_half: np.ndarray  # (K, 3)
    box_yaw: np.ndarray  # (K,)

    @property
    def n_boxes(self) -> int:
        return int(np.sum(self.box_pos[:, 2] > -0.5))


def flat_field(k: int = MAX_BOXES) -> TerrainField:
    return TerrainField(gravity=np.array([0.0, 0.0, -G]), box_pos=np.tile(PARKED_POS, (k, 1)),
                        box_half=np.tile(PARKED_HALF, (k, 1)), box_yaw=np.zeros(k))


def sample_field(xp, u, tcfg: Any, k: int = MAX_BOXES):
    """Sample a field from ``TerrainConfig`` ranges. ``u(lo, hi, shape)`` is a
    uniform sampler (numpy or jax). Shapes are fixed (K boxes) so the jax path
    is vmappable: disabled boxes are parked, not dropped.

    ``slope_share`` / ``box_share`` < 1 make the field a per-env MIXTURE: the slope (boxes)
    are switched off for a 1 - share fraction of envs. The share draws come LAST so a config
    with both shares at 1.0 consumes exactly the random stream it always did.

    Returns (gravity, box_pos, box_half, box_yaw) as ``xp`` arrays.
    """
    if has_slope(tcfg.kind):
        slope = u(np.radians(tcfg.slope_deg[0]), np.radians(tcfg.slope_deg[1]), ())
        yaw = u(np.radians(tcfg.slope_yaw_deg[0]), np.radians(tcfg.slope_yaw_deg[1]), ())
    n = tcfg.n_boxes if has_boxes(tcfg.kind) else 0
    idx = xp.arange(k)
    x = tcfg.field_start_m + idx * tcfg.box_spacing_m + u(-0.3, 0.3, (k,)) * tcfg.box_spacing_m
    y = u(-tcfg.field_width_m / 2, tcfg.field_width_m / 2, (k,))
    h = u(tcfg.box_height_m[0], tcfg.box_height_m[1], (k,))
    size = u(tcfg.box_size_m[0], tcfg.box_size_m[1], (k,))
    yaw_b = u(np.radians(tcfg.box_yaw_deg[0]), np.radians(tcfg.box_yaw_deg[1]), (k,))
    slope_share = float(getattr(tcfg, "slope_share", 1.0))
    box_share = float(getattr(tcfg, "box_share", 1.0))
    if has_slope(tcfg.kind):
        if slope_share < 1.0:
            slope = xp.where(u(0.0, 1.0, ()) < slope_share, slope, 0.0)
        gravity = gravity_for_slope(xp, slope, yaw)
    else:
        gravity = xp.asarray([0.0, 0.0, -G])
    if n and box_share < 1.0:
        n = xp.where(u(0.0, 1.0, ()) < box_share, n, 0)
    enabled = idx < n
    pos = xp.stack([x, y, PLANE_Z + h - BOX_HALF_Z], axis=-1)
    half = xp.stack([size, size, xp.full((k,), BOX_HALF_Z)], axis=-1)
    parked_pos = xp.asarray(PARKED_POS) * xp.ones((k, 1))
    parked_half = xp.asarray(PARKED_HALF) * xp.ones((k, 1))
    box_pos = xp.where(enabled[:, None], pos, parked_pos)
    box_half = xp.where(enabled[:, None], half, parked_half)
    box_yaw = xp.where(enabled, yaw_b, 0.0)
    return gravity, box_pos, box_half, box_yaw


def sample_field_numpy(rng: np.random.Generator, tcfg: Any, k: int = MAX_BOXES) -> TerrainField:
    g, p, hh, yw = sample_field(np, lambda lo, hi, shape: rng.uniform(lo, hi, shape), tcfg, k)
    return TerrainField(gravity=np.asarray(g, dtype=np.float64), box_pos=np.asarray(p, dtype=np.float64),
                        box_half=np.asarray(hh, dtype=np.float64), box_yaw=np.asarray(yw, dtype=np.float64))


def field_from_protocol(proto: dict[str, Any], k: int = MAX_BOXES) -> TerrainField:
    """A benchmark protocol's fixed terrain: scalar slope, seeded box layout."""
    kind = proto.get("kind", "flat")
    if kind not in KINDS:
        raise ValueError(f"protocol terrain kind {kind!r} not in {KINDS}")
    f = flat_field(k)
    if has_slope(kind):
        f.gravity = np.asarray(gravity_for_slope(np, np.radians(float(proto.get("slope_deg", 0.0))),
                                                 np.radians(float(proto.get("slope_yaw_deg", 0.0)))))
    if has_boxes(kind):
        from ..config import TerrainConfig  # local import: config imports nothing from here

        tcfg = TerrainConfig(kind=kind, n_boxes=int(proto.get("n_boxes", 0)),
                             box_height_m=(float(proto["box_height_m"]), float(proto["box_height_m"])),
                             box_size_m=(float(proto.get("box_size_m", 0.035)), float(proto.get("box_size_m", 0.035))),
                             box_spacing_m=float(proto.get("box_spacing_m", 0.10)),
                             box_yaw_deg=tuple(proto.get("box_yaw_deg", (-45.0, 45.0))),
                             field_start_m=float(proto.get("field_start_m", 0.15)),
                             field_width_m=float(proto.get("field_width_m", 0.40)))
        boxes = sample_field_numpy(np.random.default_rng(int(proto.get("field_seed", 0))), tcfg, k)
        f.box_pos, f.box_half, f.box_yaw = boxes.box_pos, boxes.box_half, boxes.box_yaw
    return f


def box_geom_ids(model) -> np.ndarray:
    """Geom ids of the parked terrain boxes (empty array on the flat variants)."""
    ids = []
    for i in range(MAX_BOXES):
        try:
            ids.append(model.geom(f"terrain_box_{i:02d}").id)
        except KeyError:
            break
    return np.array(ids, dtype=np.int32)


def box_body_ids(model) -> np.ndarray:
    """Body ids of the parked terrain boxes — placement goes through body_pos/body_quat (see
    build_models._add_terrain_body for why geom_pos would be invisible to collision)."""
    ids = []
    for i in range(MAX_BOXES):
        try:
            ids.append(model.body(f"terrain_box_{i:02d}").id)
        except KeyError:
            break
    return np.array(ids, dtype=np.int32)


def ground_geom_ids(model) -> np.ndarray:
    """Every geom that is ground: the floor plane plus any terrain boxes."""
    return np.concatenate([np.array([model.geom("floor").id], dtype=np.int32), box_geom_ids(model)])
