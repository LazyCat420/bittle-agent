"""The ONLY place the joint conventions live.

Three coordinate systems meet here:

* **bittle-agent degrees** — the API / viewer / moveset convention used by
  ``app/joints.py`` and ``app/motion/builtin_library.py`` (STAND = shoulders
  -45, knees 80). Rollouts are delivered to the viewer in this convention.
* **MJCF radians** — joint angles in ``static/assets/bittle.xml``. The rear
  shoulders use ``axis="0 -1 0"`` so their sign is mirrored, and the ``stand``
  keyframe knee is 85° where bittle-agent says 80°: that 5° is an *offset*,
  resolved here once (STAND must land exactly on the keyframe).
* **OpenCat firmware degrees** — gait tables; see ``opencat_gaits.py``.

Calibrated by ``trainer/tests/test_models.py``: mapping STAND through this
table must reproduce the ``stand`` keyframe within 0.1 rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Policy joint order (bittle-agent / OpenCat indices).
POLICY_JOINTS: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15)

#: bittle-agent STAND pose (from app/motion/builtin_library.py).
STAND_AGENT_DEG: dict[int, float] = {8: -45, 9: -45, 10: -45, 11: -45, 12: 80, 13: 80, 14: 80, 15: 80}


@dataclass(frozen=True)
class JointMap:
    opencat: int
    mjcf_joint: str
    ctrl_idx: int
    sign: int
    offset_deg: float
    leg: str  # rf | lf | rr | lr
    kind: str  # shoulder | knee


JOINTS: tuple[JointMap, ...] = (
    JointMap(8, "shlfs_joint", 2, +1, 0.0, "lf", "shoulder"),
    JointMap(9, "shrfs_joint", 0, +1, 0.0, "rf", "shoulder"),
    JointMap(10, "shrrs_joint", 4, -1, 0.0, "rr", "shoulder"),
    JointMap(11, "shlrs_joint", 6, -1, 0.0, "lr", "shoulder"),
    JointMap(12, "shlft_joint", 3, +1, 5.0, "lf", "knee"),
    JointMap(13, "shrft_joint", 1, +1, 5.0, "rf", "knee"),
    JointMap(14, "shrrt_joint", 5, +1, 5.0, "rr", "knee"),
    JointMap(15, "shlrt_joint", 7, +1, 5.0, "lr", "knee"),
)

BY_OPENCAT = {j.opencat: j for j in JOINTS}
BY_CTRL = {j.ctrl_idx: j for j in JOINTS}

#: ctrl index for each policy joint (policy order -> actuator order).
POLICY_TO_CTRL = np.array([BY_OPENCAT[i].ctrl_idx for i in POLICY_JOINTS], dtype=np.int32)
#: policy index for each actuator (actuator order -> policy order).
CTRL_TO_POLICY = np.argsort(POLICY_TO_CTRL).astype(np.int32)
SIGN_POLICY = np.array([BY_OPENCAT[i].sign for i in POLICY_JOINTS], dtype=np.float64)
OFFSET_POLICY_DEG = np.array([BY_OPENCAT[i].offset_deg for i in POLICY_JOINTS], dtype=np.float64)

LEGS: tuple[str, ...] = ("rf", "lf", "rr", "lr")
LEG_SHOULDER_JOINT = {"rf": "shrfs_joint", "lf": "shlfs_joint", "rr": "shrrs_joint", "lr": "shlrs_joint"}
LEG_KNEE_JOINT = {"rf": "shrft_joint", "lf": "shlft_joint", "rr": "shrrt_joint", "lr": "shlrt_joint"}
LEG_KNEE_BODY = {"rf": "servos_rf_1", "lf": "servos_lf_1", "rr": "servos_rr_1", "lr": "servos_lr_1"}
LEG_THIGH_BODY = {"rf": "c_thrf__1", "lf": "c_thlf_1", "rr": "c_thrr_1", "lr": "c_thlr_1"}
LEG_FOOT_SITE = {leg: f"{leg}_foot_site" for leg in LEGS}


def agent_deg_to_mjcf_rad(deg_policy_order: np.ndarray) -> np.ndarray:
    """bittle-agent degrees (policy order) -> MJCF radians (policy order)."""
    d = np.asarray(deg_policy_order, dtype=np.float64)
    return (d + OFFSET_POLICY_DEG) * SIGN_POLICY * math.pi / 180.0


def mjcf_rad_to_agent_deg(rad_policy_order: np.ndarray) -> np.ndarray:
    r = np.asarray(rad_policy_order, dtype=np.float64)
    return r * SIGN_POLICY * 180.0 / math.pi - OFFSET_POLICY_DEG


def policy_to_ctrl(vec_policy_order: np.ndarray) -> np.ndarray:
    """Reorder a policy-order vector into actuator (ctrl) order."""
    v = np.asarray(vec_policy_order)
    out = np.empty_like(v)
    out[..., POLICY_TO_CTRL] = v
    return out


def ctrl_to_policy(vec_ctrl_order: np.ndarray) -> np.ndarray:
    v = np.asarray(vec_ctrl_order)
    return v[..., POLICY_TO_CTRL]


def agent_pose_to_ctrl(pose: dict[int, float]) -> np.ndarray:
    """{opencat_idx: deg} (missing joints default to STAND) -> ctrl radians."""
    deg = np.array([float(pose.get(i, STAND_AGENT_DEG[i])) for i in POLICY_JOINTS])
    return policy_to_ctrl(agent_deg_to_mjcf_rad(deg))


def stand_ctrl() -> np.ndarray:
    return agent_pose_to_ctrl(STAND_AGENT_DEG)


def qpos_indices(model) -> np.ndarray:
    """qpos address of each POLICY joint, in policy order (from the compiled model)."""
    return np.array([model.jnt_qposadr[model.joint(BY_OPENCAT[i].mjcf_joint).id] for i in POLICY_JOINTS], dtype=np.int32)


def dof_indices(model) -> np.ndarray:
    return np.array([model.jnt_dofadr[model.joint(BY_OPENCAT[i].mjcf_joint).id] for i in POLICY_JOINTS], dtype=np.int32)
