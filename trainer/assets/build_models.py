"""Generate the two trainer MJCF variants from static/assets/bittle.xml.

    python -m trainer.assets.build_models

Outputs (committed; ``test_models.py`` asserts regeneration is byte-identical):

* ``generated/bittle_cpu.xml`` — full-fidelity mesh collisions (evaluation).
* ``generated/bittle_gpu.xml`` — primitive collisions only (Warp / MJX training).
* ``generated/model_report.json`` — masses, settle height, geom counts.

Common patches (both variants): fix ``meshdir``; weld the unactuated neck at
its keyframe angle; actuator ``forcerange``; drop the (always-zero) touch
sensors; add foot spheres + contact/velocity/position sensors; settle the
keyframe height. Measured masses are untouched (``inertiafromgeom=false``).
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


def _sensors(root: ET.Element, gpu: bool) -> None:
    sensor = root.find("sensor")
    for s in list(sensor):
        if s.tag == "touch":
            sensor.remove(s)
    ET.SubElement(sensor, "framezaxis", {"name": "torso_upvector", "objtype": "site", "objname": "imu_site"})
    ET.SubElement(sensor, "framelinvel", {"name": "torso_global_linvel", "objtype": "body", "objname": "torso"})
    ET.SubElement(sensor, "frameangvel", {"name": "torso_global_angvel", "objtype": "body", "objname": "torso"})
    for leg in LEGS:
        ET.SubElement(sensor, "framelinvel", {"name": f"{leg}_foot_global_linvel", "objtype": "site", "objname": LEG_FOOT_SITE[leg]})
        ET.SubElement(sensor, "framepos", {"name": f"{leg}_foot_pos", "objtype": "site", "objname": LEG_FOOT_SITE[leg]})
    for leg in LEGS:
        ET.SubElement(sensor, "contact", {"name": f"{leg}_foot_floor_found", "geom1": f"{leg}_foot", "geom2": "floor",
                                          "reduce": "mindist", "num": "1", "data": "found"})
    if gpu:
        for g in ("torso_col", "head__1_col", "jaw_1_col"):
            ET.SubElement(sensor, "contact", {"name": f"{g}_floor_found", "geom1": g, "geom2": "floor",
                                              "reduce": "mindist", "num": "1", "data": "found"})
    else:
        # mesh variant: any torso/head mesh vs floor, addressed by body
        for b in ("torso", "head__1"):
            ET.SubElement(sensor, "contact", {"name": f"{b}_floor_found", "body1": b, "geom2": "floor",
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


def build(variant: str) -> tuple[str, dict]:
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
    if variant == "gpu":
        info["mesh_collision_geoms_removed"] = _remove_mesh_collisions(root)
        info["primitives"] = _add_primitives(root)
    _sensors(root, gpu=(variant == "gpu"))
    ctrl = stand_ctrl()
    _keyframe(root, qpos, ctrl, qpos[2])

    header = (f"GENERATED by trainer/assets/build_models.py ({variant}) from static/assets/bittle.xml. DO NOT EDIT. "
              "Neck welded at keyframe; touch sensors dropped; foot spheres + contact sensors added; "
              f"servo model kp={KP} forcerange +-{FORCERANGE} N.m damping={JOINT_DAMPING}.")
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
    for variant in ("cpu", "gpu"):
        text, info = build(variant)
        (OUT_DIR / f"bittle_{variant}.xml").write_text(text)
        report["variants"][variant] = info
        print(f"wrote bittle_{variant}.xml  mass={info['mass_kg']:.4f} kg  settled_z={info['settled_z']}  ngeom={info['ngeom']}")
    (OUT_DIR / "model_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
