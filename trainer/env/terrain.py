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
* **Stairs = the same boxes, full width, one per step**, tops rising by ``rise``
  every ``tread`` along +x: a flight up, a landing, a flight down (or either half).
  Every step box is buried STAIR_BURY below the plane so there is never a gap under
  a tread, whatever its height; the parked box is baked tall enough for six 30 mm
  steps. The robot may spawn ON a step (the ``down`` profile), so both engines lift
  the spawn by the terrain height under it.

The reference height for the torso (``support_height``) is the MEAN terrain height
under the four feet, not the height under the torso point: on a staircase the point
height jumps a full riser the instant the torso centre crosses an edge, which turned
``base_height`` into a step function and tripped FALL_Z_MIN for any riser over 25 mm.
Bit-identical on flat ground (every height is 0).

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
#: LARGEST a config may request (box_size_m <= 0.15 half x; a stair step is up to 0.60 m wide and
#: 6 x 30 mm + STAIR_BURY tall): MuJoCo fixes each geom's bounding radius at compile time, so a box
#: may shrink at run time but never grow past its compiled size. Must match
#: build_models.TERRAIN_PARKED_SIZE.
PARKED_POS = np.array([0.0, 0.0, -1.0])
PARKED_HALF = np.array([0.15, 0.30, 0.10])
#: Half height of an enabled ROCK; the box top sits at PLANE_Z + protrusion.
BOX_HALF_Z = 0.02
#: A stair step's box bottom sits this far BELOW the plane (no gap under any tread).
STAIR_BURY = 0.02
#: Steps per flight a config may ask for; the field holds up + landing + down = 2 * MAX + 1 slots.
MAX_STAIR_STEPS = 6
STAIR_LANDING_SLOT = MAX_STAIR_STEPS
STAIR_PROFILES = ("up", "down", "up_down")
G = 9.81
KINDS = ("flat", "slope", "rough", "rough_slope", "stairs")
#: Body-frame (x, y) offsets of the critic's 3x3 height scan.
SCAN_OFFSETS = np.array([(dx, dy) for dx in (-0.06, 0.0, 0.06) for dy in (-0.06, 0.0, 0.06)])
#: Size of the critic-only terrain block appended to ``privileged_state``.
PRIVILEGED_TERRAIN_DIM = len(SCAN_OFFSETS) + 4 + 3


def has_boxes(kind: str) -> bool:
    return kind in ("rough", "rough_slope", "stairs")


def has_stairs(kind: str) -> bool:
    return kind == "stairs"


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
    # a PARKED box (1 m below the floor, 0.15 m half-extents around the origin) must never count as
    # "under" the point: with every box parked, max() over the parked tops returned -0.97 m for any
    # point within 0.15 m of the spawn and the height-relative reward, clearance and critic scan all
    # read the robot as a metre in the air (found 2026-09-08 on the first box_share < 1 run)
    inside = (xp.abs(u) <= box_half[:, 0]) & (xp.abs(v) <= box_half[:, 1]) & (box_pos[:, 2] > PARKED_POS[2] + 0.5)
    top = box_pos[:, 2] + box_half[:, 2] - PLANE_Z
    return xp.maximum(xp.max(xp.where(inside, top, 0.0)), 0.0)


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


def support_height(xp, foot_geom_pos, box_pos, box_half, box_yaw):
    """Mean terrain height under the four feet: the torso's height reference for ``base_height``,
    fall termination and body clearance. Continuous across a stair edge (front feet on the step
    above, rear feet below -> half a riser), exactly 0 on flat ground and where every box is parked."""
    hs = [terrain_height(xp, foot_geom_pos[i, 0], foot_geom_pos[i, 1], box_pos, box_half, box_yaw)
          for i in range(foot_geom_pos.shape[0])]
    return sum(hs) / len(hs)


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


def field_to_json(field: TerrainField) -> dict[str, Any]:
    """The field as the 3D viewer draws it: enabled boxes in MuJoCo world metres (x forward, y left,
    z up; the floor plane sits at PLANE_Z), plus the gravity vector so a tilted-world incline can be
    shown by tilting the camera's up-vector."""
    g = np.asarray(field.gravity, dtype=float)
    gn = g / max(float(np.linalg.norm(g)), 1e-9)
    slope = float(np.degrees(np.arccos(np.clip(-gn[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(-gn[1], -gn[0]))) if slope > 1e-3 else 0.0
    boxes = []
    for i in range(int(field.box_pos.shape[0])):
        if field.box_pos[i, 2] <= -0.5:
            continue
        boxes.append({"pos": [round(float(v), 5) for v in field.box_pos[i]],
                      "half": [round(float(v), 5) for v in field.box_half[i]],
                      "yaw_deg": round(float(np.degrees(field.box_yaw[i])), 2)})
    return {"gravity": [round(float(v), 5) for v in g], "slope_deg": round(slope, 3), "slope_yaw_deg": round(yaw, 2),
            "plane_z": PLANE_Z, "boxes": boxes}


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
    if has_stairs(tcfg.kind):
        return sample_stairs(xp, u, tcfg, k)
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


def sample_stairs(xp, u, tcfg: Any, k: int = MAX_BOXES):
    """A staircase along +x from ``field_start_m``: ``n`` steps up (slots 0..MAX-1), a landing (slot MAX),
    ``n`` steps down (slots MAX+1..2*MAX), the rest parked. ``n``, rise and tread are drawn per env from
    ``stair_steps`` / ``stair_rise_m`` / ``stair_tread_m``; the profile is static:

    * ``up``       — the flight up and the landing (the robot ends on top)
    * ``down``     — the landing is a PLATFORM ending at ``field_start_m`` (the robot spawns on it) and the
                     flight descends from there
    * ``up_down``  — up, landing, down (the default: one episode teaches both)

    Slot layout is fixed so the jax path is vmappable; ``box_share`` < 1 parks the whole staircase for
    a 1 - share fraction of envs (a stairs/flat mixture). Gravity is flat: stairs are level.
    """
    assert k >= 2 * MAX_STAIR_STEPS + 1
    lo_n, hi_n = int(tcfg.stair_steps[0]), int(tcfg.stair_steps[1])
    n = xp.floor(u(float(lo_n), float(hi_n) + 1.0 - 1e-6, ()))
    n = xp.clip(n, lo_n, hi_n)
    rise = u(tcfg.stair_rise_m[0], tcfg.stair_rise_m[1], ())
    tread = u(tcfg.stair_tread_m[0], tcfg.stair_tread_m[1], ())
    landing = float(tcfg.stair_landing_m)
    half_w = float(tcfg.stair_width_m) / 2.0
    profile = str(tcfg.stair_profile)
    start = float(tcfg.field_start_m)
    idx = xp.arange(k)
    slot = idx * 1.0
    # ── tops and x-centres per slot ──
    up_i = slot                                   # slots 0..MAX-1: step i (0-based)
    down_j = slot - (MAX_STAIR_STEPS + 1)          # slots MAX+1..: step j of the descent
    if profile == "down":
        flight_x0 = start                          # the descent starts at field_start
        land_c = start - landing / 2.0             # the platform sits BEHIND field_start (under the spawn)
        up_on = xp.zeros((k,), dtype=bool)
    else:
        flight_x0 = start + n * tread + landing    # descent starts after the flight up + landing
        land_c = start + n * tread + landing / 2.0
        up_on = idx < n
    land_on = idx == STAIR_LANDING_SLOT
    down_on = (idx > STAIR_LANDING_SLOT) & (down_j < n) if profile != "up" else xp.zeros((k,), dtype=bool)
    top = xp.where(up_on, (up_i + 1.0) * rise,
                   xp.where(land_on, n * rise,
                            xp.where(down_on, (n - 1.0 - down_j) * rise, 0.0)))
    cx = xp.where(up_on, start + (up_i + 0.5) * tread,
                  xp.where(land_on, land_c,
                           xp.where(down_on, flight_x0 + (down_j + 0.5) * tread, 0.0)))
    hx = xp.where(land_on, landing / 2.0, tread / 2.0)
    enabled = up_on | land_on | down_on
    enabled = enabled & (top > 1e-6)              # a 0-step flight has no boxes at all
    box_share = float(getattr(tcfg, "box_share", 1.0))
    if box_share < 1.0:
        enabled = enabled & (u(0.0, 1.0, ()) < box_share)
    half_z = (top + STAIR_BURY) / 2.0             # bottom at PLANE_Z - STAIR_BURY, whatever the top
    cz = PLANE_Z + top - half_z
    pos = xp.stack([cx, xp.zeros((k,)), cz], axis=-1)
    half = xp.stack([hx, xp.full((k,), half_w), half_z], axis=-1)
    parked_pos = xp.asarray(PARKED_POS) * xp.ones((k, 1))
    parked_half = xp.asarray(PARKED_HALF) * xp.ones((k, 1))
    box_pos = xp.where(enabled[:, None], pos, parked_pos)
    box_half = xp.where(enabled[:, None], half, parked_half)
    box_yaw = xp.zeros((k,))
    gravity = xp.asarray([0.0, 0.0, -G])
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
    if has_stairs(kind):
        from ..config import TerrainConfig  # local import: config imports nothing from here

        n = int(proto.get("stair_steps", 3))
        tcfg = TerrainConfig(kind="stairs", stair_steps=(n, n),
                             stair_rise_m=(float(proto["stair_rise_m"]), float(proto["stair_rise_m"])),
                             stair_tread_m=(float(proto.get("stair_tread_m", 0.08)), float(proto.get("stair_tread_m", 0.08))),
                             stair_profile=str(proto.get("stair_profile", "up_down")),
                             stair_landing_m=float(proto.get("stair_landing_m", 0.30)),
                             stair_width_m=float(proto.get("stair_width_m", 0.60)),
                             field_start_m=float(proto.get("field_start_m", 0.15)))
        stairs = sample_field_numpy(np.random.default_rng(int(proto.get("field_seed", 0))), tcfg, k)
        f.box_pos, f.box_half, f.box_yaw = stairs.box_pos, stairs.box_half, stairs.box_yaw
        return f
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
