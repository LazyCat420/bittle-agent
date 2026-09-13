"""Generate bittle_basketball.xml for the dynamic basketball balance benchmark.

Constructs the MuJoCo simulation model with:
- Floor at z = -0.01 m (Plane).
- Hollow thin-shell spherical basketball:
  * Radius: r = 0.12 m
  * Mass: m = 0.60 kg
  * Inertia: I = 2/3 * m * r^2 = 0.00576 kg.m^2 (thin spherical shell)
  * MuJoCo condim=6 (sliding friction 1.0, torsional 0.005, rolling 0.005)
  * Contact compliance: solref=[0.015, 1.0], solimp=[0.9, 0.95, 0.001]
- Petoi Bittle spawned atop the sphere with 4 paws contacting the curved surface.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GENERATED_DIR = HERE / "generated"
CPU_XML_PATH = GENERATED_DIR / "bittle_cpu.xml"
OUT_XML_PATH = GENERATED_DIR / "bittle_basketball.xml"

BALL_RADIUS = 0.12       # m
BALL_MASS = 0.60         # kg
BALL_INERTIA = (2.0 / 3.0) * BALL_MASS * (BALL_RADIUS ** 2)  # 0.00576 kg.m^2
BALL_Z_CENTER = -0.01 + BALL_RADIUS  # 0.11 m (rests on floor at z = -0.01)

LEGS = ("lf", "rf", "lr", "rr")


def build_basketball_xml() -> str:
    if not CPU_XML_PATH.exists():
        from .build_models import main as build_all
        build_all()

    raw = CPU_XML_PATH.read_text()
    root = ET.fromstring(raw)

    # 1. Add basketball material in asset if not present
    asset = root.find("asset")
    if asset is not None and root.find(".//material[@name='basketball_mat']") is None:
        ET.SubElement(asset, "material", {
            "name": "basketball_mat",
            "rgba": "0.95 0.40 0.10 1.0",
            "specular": "0.3",
            "shininess": "0.2",
        })

    # 2. Add basketball body into worldbody
    worldbody = root.find("worldbody")
    assert worldbody is not None, "Missing worldbody"

    ball_body = ET.SubElement(worldbody, "body", {
        "name": "basketball",
        "pos": f"0 0 {BALL_Z_CENTER:.6f}",
    })
    ET.SubElement(ball_body, "freejoint", {"name": "basketball_free"})
    ET.SubElement(ball_body, "inertial", {
        "pos": "0 0 0",
        "mass": f"{BALL_MASS:.4f}",
        "diaginertia": f"{BALL_INERTIA:.6f} {BALL_INERTIA:.6f} {BALL_INERTIA:.6f}",
    })
    ET.SubElement(ball_body, "geom", {
        "name": "basketball_geom",
        "type": "sphere",
        "size": f"{BALL_RADIUS:.4f}",
        "material": "basketball_mat",
        "condim": "6",
        "friction": "1.0 0.005 0.005",
        "solref": "0.015 1.0",
        "solimp": "0.9 0.95 0.001",
    })

    # 3. Add contact sensors for foot-to-ball and torso-to-ball
    sensor = root.find("sensor")
    if sensor is not None:
        for leg in LEGS:
            ET.SubElement(sensor, "contact", {
                "name": f"{leg}_ball_found",
                "geom1": f"{leg}_foot",
                "geom2": "basketball_geom",
                "reduce": "mindist",
                "num": "1",
                "data": "found",
            })
        for body_part in ("torso", "head__1", "jaw_1"):
            ET.SubElement(sensor, "contact", {
                "name": f"{body_part}_ball_found",
                "body1": body_part,
                "geom2": "basketball_geom",
                "reduce": "mindist",
                "num": "1",
                "data": "found",
            })

    # 4. Update keyframe
    # When a freejoint is added to worldbody, nq increases by 7 (pos 3 + quat 4), nv by 6.
    # MuJoCo orders free joints in body definition order.
    # Torso is defined first, basketball is defined second.
    key = root.find("keyframe").find("key")
    if key is not None:
        qpos_vals = [float(x) for x in key.get("qpos").split()]
        # Adjust robot torso x and z to center over ball:
        # Bittle center of foot stance is x ≈ -0.0231, so shifting torso x to +0.0231 centers feet at x=0
        qpos_vals[0] = 0.0231
        qpos_vals[1] = 0.0
        # Elevate torso: ball top is 0.23 m, torso settled height on flat ground is 0.047 m,
        # with foot drop on sphere curve (~0.023 m), initial torso z is ~0.254 m
        qpos_vals[2] = 0.254

        # Append basketball freejoint qpos: (x, y, z, qw, qx, qy, qz)
        ball_qpos = [0.0, 0.0, BALL_Z_CENTER, 1.0, 0.0, 0.0, 0.0]
        full_qpos = qpos_vals + ball_qpos
        key.set("qpos", " ".join(f"{x:.6g}" for x in full_qpos))

    ET.indent(root, space="  ")
    xml_text = ET.tostring(root, encoding="unicode")
    header = (
        "<!-- GENERATED for Bittle Basketball Dynamic Balancing Benchmark. "
        f"Thin shell inertia I={BALL_INERTIA:.6f} kg.m^2, condim=6 rolling friction. -->\n"
    )
    return header + xml_text


def main() -> int:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    xml_str = build_basketball_xml()
    OUT_XML_PATH.write_text(xml_str)
    print(f"Generated {OUT_XML_PATH} successfully.")

    # Settle test using MuJoCo
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml_str, assets={p.name: p.read_bytes() for p in (REPO / "static" / "assets" / "meshes").glob("*.obj")})
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    print(f"Model compiled: nq={m.nq}, nv={m.nv}, ngeom={m.ngeom}, nbody={m.nbody}")
    
    # Run 500 settle steps with joint position control
    for _ in range(500):
        mujoco.mj_step(m, d)
        
    print(f"Settle completed: ball_pos={d.qpos[m.nq-7:m.nq-4]}, robot_torso_pos={d.qpos[0:3]}, ncon={d.ncon}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
