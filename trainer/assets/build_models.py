"""Generate the trainer MJCF variants from static/assets/bittle.xml.

    python -m trainer.assets.build_models

Outputs (committed; ``test_models.py`` asserts regeneration is byte-identical):

* ``generated/bittle_cpu.xml`` — full-fidelity mesh collisions (evaluation).
* ``generated/bittle_gpu.xml`` — primitive collisions only (Warp / MJX training).
* ``generated/bittle_{cpu,gpu}_terrain.xml`` — the same plus a static ``terrain``
  body holding the floor and 32 parked boxes (rocks / edges, placed per env).
* ``generated/model_report.json`` — masses, settle height, geom counts.

Common patches (all variants): fix ``meshdir``; weld the unactuated neck at
its keyframe angle; actuator ``forcerange``; drop the (always-zero) touch
sensors; add foot spheres + contact/velocity/position sensors; settle the
keyframe height. Measured masses are untouched (``inertiafromgeom=false``).

The mesh (cpu) variants also EXCLUDE contacts between the chassis meshes and
the knee-servo / shank bodies: MuJoCo collides the convex hull of each mesh,
and the chassis hull fills the concave underside where the legs tuck, so the
firmware ``wkF`` / ``crF`` gaits — which the real robot walks — penetrate the
chassis in 64/116 and 103/103 frames (up to 20 mm). A leg pushed into that
phantom wall pins its shoulder servo at the torque cap: a stalled servo.
"""

from __future__ import annotations

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .joint_map import LEG_FOOT_SITE, LEG_KNEE_BODY, LEG_THIGH_BODY, LEGS, STAND_AGENT_DEG, stand_ctrl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SOURCE_XML = REPO / "static" / "assets" / "bittle.xml"
MESH_DIR = REPO / "static" / "assets" / "meshes"
OUT_DIR = HERE / "generated"

# Servo model (P1S: ~2.5 kg.cm stall, 0.11 s/60deg). The source MJCF used kp=40 with
# joint damping 1.5 and NO torque limit, which only works because the actuator can
# pull several N.m; with a realistic torque cap those joints cannot move at all
# (measured: trot replay 0.000 m). So the trainer variants use a stiff but
# torque-limited servo with light joint damping instead. Nominal values below,
# randomised by DR (trainer/config.py DRConfig).
KP = 10.0            # N.m/rad  -> saturates at ~1.4 deg error (servo-like)
FORCERANGE = 0.25    # N.m      -> P1S stall torque (~2.5 kg.cm)
JOINT_DAMPING = 0.05  # N.m.s/rad
JOINT_ARMATURE = 0.005
JOINT_FRICTIONLOSS = 0.01
FOOT_RADIUS = 0.006
THIGH_RADIUS = 0.008
SHANK_RADIUS = 0.005
SHANK_FRACTION = 0.75  # capsule stops short of the foot sphere so the sphere touches down first
TORSO_MESHES = ("base_link", "front__1", "rear__1", "cover_1", "battery_1",
                "servo_rfs_1", "servo_rrs__1", "servo_lfs_1", "servo_lrs__1")
#: chassis bodies whose hulls over-collide with the tucked legs (cpu variants)
CHASSIS_BODIES = ("torso", "front__1", "rear__1", "cover_1", "battery_1")
LEG_TUCK_BODIES = tuple(LEG_KNEE_BODY[leg] for leg in LEGS) + tuple(f"shank_{leg}_1" for leg in LEGS)
TERRAIN_MAX_BOXES = 32
TERRAIN_PARKED_POS = "0 0 -1"
#: Parked boxes are baked at the LARGEST size a config may ask for (TerrainConfig.box_size_m <= 0.15 half x,
#: a stair step 0.60 m wide, six 35 mm risers + the 20 mm burial = 0.10 half z; must equal
#: terrain.PARKED_HALF): MuJoCo computes each geom's bounding radius / AABB at compile time and the
#: broadphase never revisits it, so a box grown at run time beyond its compiled size is silently
#: skipped by collision detection (measured: zero foot-box contacts with 1 mm parked boxes). Shrinking
#: is always safe. Parked 1 m below the floor they cannot touch anything at any size.
TERRAIN_PARKED_SIZE = "0.15 0.30 0.10"


# ── helpers ────────────────────────────────────────────────────────────────

def _f(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=np.float64)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), a[0] * s, a[1] * s, a[2] * s])


def _obj_vertices(name: str, scale: float = 0.001) -> np.ndarray:
    pts = []
    with open(MESH_DIR / f"{name}.obj") as fh:
        for line in fh:
            if line.startswith("v "):
                pts.append([float(x) for x in line.split()[1:4]])
    return np.array(pts) * scale


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {c: p for p in root.iter() for c in p}


def _body_chain_offset(root: ET.Element, ancestor: str, descendant: str) -> np.ndarray:
    """Translation of ``descendant`` body origin expressed in ``ancestor`` frame (no quats on the chain)."""
    parents = _parent_map(root)
    b = _find_body(root, descendant)
    off = np.zeros(3)
    while b is not None and b.get("name") != ancestor:
        if b.get("quat") not in (None, "1 0 0 0"):
            raise ValueError(f"body {b.get('name')} has a rotation; chain offset needs quats handled")
        off += _f(b.get("pos", "0 0 0"))
        b = parents.get(b)
        if b is None or b.tag != "body":
            raise KeyError(f"{ancestor} is not an ancestor of {descendant}")
    return off


def _mesh_aabb_in_body(root: ET.Element, body_name: str, mesh_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """AABB over the visual meshes found under ``body_name`` (recursively, no joints crossed)."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    body = _find_body(root, body_name)
    parents = _parent_map(root)
    for g in body.iter("geom"):
        if g.get("mesh") not in mesh_names or g.get("class") != "visual":
            continue
        # accumulate body offsets from g's body up to body_name
        off = np.zeros(3)
        b = parents[g]
        while b is not body:
            off += _f(b.get("pos", "0 0 0"))
            b = parents[b]
        pts = _obj_vertices(g.get("mesh")) + _f(g.get("pos", "0 0 0")) + off
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
    return lo, hi


# ── patches ────────────────────────────────────────────────────────────────

def _fix_compiler(root: ET.Element) -> None:
    comp = root.find("compiler")
    comp.set("meshdir", "../../../static/assets/meshes")


def _weld_neck(root: ET.Element, keyframe_qpos: list[float]) -> list[float]:
    """Replace neck_joint by a fixed rotation at the keyframe angle. Returns qpos without the neck."""
    body = _find_body(root, "servo_neck__1")
    joint = next(j for j in body.findall("joint") if j.get("name") == "neck_joint")
    axis = _f(joint.get("axis"))
    angle = keyframe_qpos[7]  # qpos order: root(7), neck, ...
    body.set("quat", _fmt(_axis_angle_quat(axis, angle)))
    body.remove(joint)
    return keyframe_qpos[:7] + keyframe_qpos[8:]


def _actuator_forcerange(root: ET.Element) -> None:
    default = root.find("default")
    pos_default = default.find("position")
    pos_default.set("forcerange", f"-{FORCERANGE} {FORCERANGE}")
    pos_default.set("kp", f"{KP}")
    joint_default = default.find("joint")
    joint_default.set("damping", f"{JOINT_DAMPING}")
    joint_default.set("armature", f"{JOINT_ARMATURE}")
    joint_default.set("frictionloss", f"{JOINT_FRICTIONLOSS}")


def _add_prim_class(root: ET.Element) -> None:
    default = root.find("default")
    cls = ET.SubElement(default, "default", {"class": "prim"})
    ET.SubElement(cls, "geom", {
        "contype": "1", "conaffinity": "0", "condim": "3",
        "friction": "0.9 0.02 0.01", "group": "4", "rgba": "1 0.2 0.2 0.35",
    })


def _foot_centers(root: ET.Element) -> dict[str, np.ndarray]:
    """Sphere centre per leg, in the knee (servos_*) body frame.

    The foot site sits at the FK tip (z=0 at stand) but the rubber paw mesh
    reaches ~10 mm lower. Compile the model as-is, pose it at the keyframe,
    find the lowest shank-mesh vertex in world, and put the sphere centre one
    radius above it so the primitive foot touches the floor exactly where the
    mesh does. Computed once from the unmodified tree so both variants agree.
    """
    import mujoco

    text = _serialize(root, "tmp")
    m = mujoco.MjModel.from_xml_string(text, assets=_assets())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    parents = _parent_map(root)
    out = {}
    for leg in LEGS:
        knee_name = LEG_KNEE_BODY[leg]
        knee = _find_body(root, knee_name)
        shank = next(b for b in knee.findall("body") if b.get("name", "").startswith("shank_"))
        g = next(g for g in shank.findall("geom") if g.get("class") == "visual")
        pts = _obj_vertices(g.get("mesh")) + _f(g.get("pos", "0 0 0")) + _f(shank.get("pos", "0 0 0"))
        bid = m.body(knee_name).id
        R = d.xmat[bid].reshape(3, 3)
        world = pts @ R.T + d.xpos[bid]
        lowest = world[np.argmin(world[:, 2])]
        centre_world = lowest + np.array([0.0, 0.0, FOOT_RADIUS])
        out[leg] = R.T @ (centre_world - d.xpos[bid])
    return out


def _add_feet(root: ET.Element, centres: dict[str, np.ndarray]) -> None:
    for leg in LEGS:
        body = _find_body(root, LEG_KNEE_BODY[leg])
        ET.SubElement(body, "geom", {
            "name": f"{leg}_foot", "type": "sphere", "size": f"{FOOT_RADIUS}",
            "pos": _fmt(centres[leg]), "class": "prim",
        })


def _remove_mesh_collisions(root: ET.Element) -> int:
    n = 0
    for body in root.iter("body"):
        for g in list(body.findall("geom")):
            if g.get("class") == "collision":
                body.remove(g)
                n += 1
    return n


def _add_primitives(root: ET.Element) -> dict:
    info = {}
    # torso box
    lo, hi = _mesh_aabb_in_body(root, "torso", TORSO_MESHES)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    torso = _find_body(root, "torso")
    ET.SubElement(torso, "geom", {"name": "torso_col", "type": "box", "pos": _fmt(center), "size": _fmt(half), "class": "prim"})
    info["torso_box"] = {"center": center.tolist(), "half": half.tolist()}
    # head + jaw boxes (in their own bodies; the neck is welded at the keyframe angle)
    for bname, mesh in (("head__1", "head__1"), ("jaw_1", "jaw_1")):
        lo, hi = _mesh_aabb_in_body(root, bname, (mesh,))
        center, half = (lo + hi) / 2, (hi - lo) / 2
        ET.SubElement(_find_body(root, bname), "geom", {
            "name": f"{bname}_col", "type": "box", "pos": _fmt(center), "size": _fmt(half), "class": "prim"})
        info[f"{bname}_box"] = {"center": center.tolist(), "half": half.tolist()}
    # legs: thigh capsule shoulder->knee, shank capsule knee->foot
    for leg in LEGS:
        thigh = _find_body(root, LEG_THIGH_BODY[leg])
        knee_off = _body_chain_offset(root, LEG_THIGH_BODY[leg], LEG_KNEE_BODY[leg])
        ET.SubElement(thigh, "geom", {
            "name": f"{leg}_thigh_col", "type": "capsule", "size": f"{THIGH_RADIUS}",
            "fromto": _fmt(np.concatenate([np.zeros(3), knee_off])), "class": "prim"})
        knee = _find_body(root, LEG_KNEE_BODY[leg])
        foot = next(g for g in knee.findall("geom") if g.get("name") == f"{leg}_foot")
        ET.SubElement(knee, "geom", {
            "name": f"{leg}_shank_col", "type": "capsule", "size": f"{SHANK_RADIUS}",
            "fromto": _fmt(np.concatenate([np.zeros(3), _f(foot.get("pos")) * SHANK_FRACTION])), "class": "prim"})
        info[f"{leg}_thigh_vec"] = knee_off.tolist()
    return info


def _exclude_chassis_leg_contacts(root: ET.Element) -> int:
    """Mesh variants: the chassis hulls must not collide with the tucked knee/shank bodies."""
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    n = 0
    for chassis in CHASSIS_BODIES:
        for leg_body in LEG_TUCK_BODIES:
            ET.SubElement(contact, "exclude", {"body1": chassis, "body2": leg_body})
            n += 1
    return n


def _add_terrain_class(root: ET.Element) -> None:
    default = root.find("default")
    cls = ET.SubElement(default, "default", {"class": "terrain"})
    # conaffinity=1 so the robot's prim class (contype 1, conaffinity 0) collides with it; margin 0.
    # group 2: visible by default (the renderer hides groups >= 3, which made the first rough GIFs look flat)
    ET.SubElement(cls, "geom", {
        "type": "box", "contype": "1", "conaffinity": "1", "condim": "3",
        "friction": "0.9 0.02 0.01", "group": "2", "rgba": "0.55 0.48 0.40 1",
    })


def _add_terrain_body(root: ET.Element) -> None:
    """Move ``floor`` into a static body ``terrain`` whose CHILD bodies each carry one parked box.

    Why one body per box, placed through ``body_pos``/``body_quat`` and never ``geom_pos``:
    MuJoCo builds a per-body BVH over each body's geoms at compile time, in the body frame, and
    the broadphase trusts it forever. A geom moved at run time via ``geom_pos`` keeps its
    compile-time AABB, so a box "moved" from its parking spot at z = -1 is never even tested
    for collision (measured: zero foot-box contacts, while mj_ray — which reads geom_pos — saw
    it). A body moved via ``body_pos`` is re-posed by mj_kinematics every step, and its
    one-geom BVH (geom at the body origin, compile-time MAX size) stays valid. Contact sensors
    key on ``subtree2="terrain"`` so floor and boxes are one ground.
    """
    world = root.find("worldbody")
    floor = next(g for g in world.findall("geom") if g.get("name") == "floor")
    world.remove(floor)
    body = ET.Element("body", {"name": "terrain"})
    body.append(floor)
    for i in range(TERRAIN_MAX_BOXES):
        b = ET.SubElement(body, "body", {"name": f"terrain_box_{i:02d}", "pos": TERRAIN_PARKED_POS})
        ET.SubElement(b, "geom", {"name": f"terrain_box_{i:02d}", "class": "terrain", "size": TERRAIN_PARKED_SIZE})
    world.insert(0, body)


def _name_knee_collision_meshes(root: ET.Element) -> None:
    """Mesh variants: name the knee-servo housing's collision mesh so a stumble sensor can address it
    (the knee body also carries the foot sphere, and the shank mesh includes the rubber paw)."""
    for leg in LEGS:
        knee = _find_body(root, LEG_KNEE_BODY[leg])
        g = next(g for g in knee.findall("geom") if g.get("class") == "collision")
        g.set("name", f"{leg}_knee_col")


def _sensors(root: ET.Element, gpu: bool, terrain: bool = False) -> None:
    sensor = root.find("sensor")
    for s in list(sensor):
        if s.tag == "touch":
            sensor.remove(s)
    # ground reference: the floor geom on the flat variants, the terrain SUBTREE (floor + box bodies) otherwise
    ground = {"subtree2": "terrain"} if terrain else {"geom2": "floor"}
    ET.SubElement(sensor, "framezaxis", {"name": "torso_upvector", "objtype": "site", "objname": "imu_site"})
    ET.SubElement(sensor, "framelinvel", {"name": "torso_global_linvel", "objtype": "body", "objname": "torso"})
    ET.SubElement(sensor, "frameangvel", {"name": "torso_global_angvel", "objtype": "body", "objname": "torso"})
    for leg in LEGS:
        ET.SubElement(sensor, "framelinvel", {"name": f"{leg}_foot_global_linvel", "objtype": "site", "objname": LEG_FOOT_SITE[leg]})
        ET.SubElement(sensor, "framepos", {"name": f"{leg}_foot_pos", "objtype": "site", "objname": LEG_FOOT_SITE[leg]})
    for leg in LEGS:
        ET.SubElement(sensor, "contact", {"name": f"{leg}_foot_floor_found", "geom1": f"{leg}_foot", **ground,
                                          "reduce": "mindist", "num": "1", "data": "found"})
    if gpu:
        for g in ("torso_col", "head__1_col", "jaw_1_col"):
            ET.SubElement(sensor, "contact", {"name": f"{g}_floor_found", "geom1": g, **ground,
                                              "reduce": "mindist", "num": "1", "data": "found"})
    else:
        # mesh variant: any torso/head mesh vs floor, addressed by body
        for b in ("torso", "head__1"):
            ET.SubElement(sensor, "contact", {"name": f"{b}_floor_found", "body1": b, **ground,
                                              "reduce": "mindist", "num": "1", "data": "found"})
    if terrain:
        # stumble sensors: shank / thigh on the ground (reward term, NOT fall termination)
        for leg in LEGS:
            # gpu: shank capsule (stops 75% of the way to the foot) / thigh capsule.
            # cpu: the knee-servo housing mesh (the shank mesh includes the rubber paw, which touches
            # the ground on every stance) / the thigh bracket body.
            for part, gpu_ref, cpu_ref in (("shank", {"geom1": f"{leg}_shank_col"}, {"geom1": f"{leg}_knee_col"}),
                                           ("thigh", {"geom1": f"{leg}_thigh_col"}, {"body1": LEG_THIGH_BODY[leg]})):
                ref = gpu_ref if gpu else cpu_ref
                ET.SubElement(sensor, "contact", {"name": f"{leg}_{part}_floor_found", **ref, **ground,
                                                  "reduce": "mindist", "num": "1", "data": "found"})


def _keyframe(root: ET.Element, qpos: list[float], ctrl: np.ndarray, z: float) -> None:
    key = root.find("keyframe").find("key")
    qpos = list(qpos)
    qpos[2] = z
    key.set("qpos", _fmt(qpos))
    key.set("ctrl", _fmt(ctrl))


def _serialize(root: ET.Element, header: str) -> str:
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return f"<!-- {header} -->\n{body}\n"


# ── build ──────────────────────────────────────────────────────────────────

def _settle_height(xml_text: str, base_dir: Path) -> tuple[float, dict]:
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml_text, assets=_assets())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    feet = {leg: d.site(LEG_FOOT_SITE[leg]).xpos.tolist() for leg in LEGS}
    return float(d.qpos[2]), {"feet": feet, "ncon": int(d.ncon),
                              "mass_kg": float(m.body_subtreemass[m.body("torso").id]),
                              "ngeom": int(m.ngeom), "nq": int(m.nq), "nu": int(m.nu)}


def _assets() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in MESH_DIR.glob("*.obj")}


VARIANTS = ("cpu", "gpu", "cpu_terrain", "gpu_terrain")


def build(variant: str) -> tuple[str, dict]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    gpu = variant.startswith("gpu")
    terrain = variant.endswith("_terrain")
    # MuJoCo tolerates '--' inside XML comments; ElementTree does not, so strip comments first.
    raw = re.sub(r"<!--.*?-->", "", SOURCE_XML.read_text(), flags=re.S)
    root = ET.fromstring(raw)
    key = root.find("keyframe").find("key")
    qpos = [float(x) for x in key.get("qpos").split()]

    _fix_compiler(root)
    qpos = _weld_neck(root, qpos)
    _keyframe(root, qpos, stand_ctrl(), qpos[2])  # keyframe must match nq before any compile
    _actuator_forcerange(root)
    _add_prim_class(root)
    centres = _foot_centers(root)
    _add_feet(root, centres)
    info_feet = {leg: [round(float(x), 6) for x in c] for leg, c in centres.items()}
    info: dict = {"variant": variant, "foot_sphere_centres": info_feet}
    if gpu:
        info["mesh_collision_geoms_removed"] = _remove_mesh_collisions(root)
        info["primitives"] = _add_primitives(root)
    else:
        info["chassis_leg_contact_excludes"] = _exclude_chassis_leg_contacts(root)
        _name_knee_collision_meshes(root)
    if terrain:
        _add_terrain_class(root)
        _add_terrain_body(root)
        info["terrain_boxes"] = TERRAIN_MAX_BOXES
    _sensors(root, gpu=gpu, terrain=terrain)
    ctrl = stand_ctrl()
    _keyframe(root, qpos, ctrl, qpos[2])

    header = (f"GENERATED by trainer/assets/build_models.py ({variant}) from static/assets/bittle.xml. DO NOT EDIT. "
              "Neck welded at keyframe; touch sensors dropped; foot spheres + contact sensors added; "
              f"servo model kp={KP} forcerange +-{FORCERANGE} N.m damping={JOINT_DAMPING}"
              + ("; chassis-vs-tucked-leg contacts excluded" if not gpu else "")
              + (f"; terrain body with {TERRAIN_MAX_BOXES} parked boxes" if terrain else "") + ".")
    text = _serialize(root, header)
    z, stats = _settle_height(text, OUT_DIR)
    _keyframe(root, qpos, ctrl, round(z, 5))
    text = _serialize(root, header)
    z2, stats = _settle_height(text, OUT_DIR)
    info.update(stats)
    info["settled_z"] = round(z2, 5)
    info["keyframe_z"] = round(z, 5)
    info["stand_ctrl"] = [round(float(c), 5) for c in ctrl]
    info["stand_agent_deg"] = STAND_AGENT_DEG
    return text, info


def main(argv: list[str] | None = None) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"source": str(SOURCE_XML.relative_to(REPO)), "variants": {}}
    for variant in VARIANTS:
        text, info = build(variant)
        (OUT_DIR / f"bittle_{variant}.xml").write_text(text)
        report["variants"][variant] = info
        print(f"wrote bittle_{variant}.xml  mass={info['mass_kg']:.4f} kg  settled_z={info['settled_z']}  ngeom={info['ngeom']}")
    (OUT_DIR / "model_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
