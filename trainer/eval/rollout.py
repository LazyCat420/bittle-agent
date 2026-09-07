"""``bittle.rollout.v1`` recorder: what the 3D viewer plays back."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..assets.joint_map import POLICY_JOINTS

SCHEMA = "bittle.rollout.v1"


class RolloutRecorder:
    def __init__(self, fps: float, source: dict[str, Any]):
        self.fps = fps
        self.source = source
        self.frames: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def record(self, t: float, env, applied_deg: np.ndarray, contact: np.ndarray, cmd: np.ndarray) -> None:
        base = env.base_state()
        self.frames.append({
            "t": round(float(t), 4),
            "angles_deg": [int(round(float(x))) for x in applied_deg],
            "measured_deg": [round(float(x), 1) for x in base["joint_deg"]],
            "base_pos_m": [round(float(x), 5) for x in base["pos"]],
            "base_quat_wxyz": [round(float(x), 5) for x in base["quat_wxyz"]],
            "base_rpy_deg": [round(float(x), 2) for x in base["rpy_deg"]],
            "contacts": [int(bool(c)) for c in contact],
            "cmd": [round(float(x), 3) for x in cmd],
        })

    def event(self, t: float, kind: str, **extra: Any) -> None:
        self.events.append({"t": round(float(t), 4), "type": kind, **extra})

    def to_dict(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "fps": self.fps,
            "dt": 1.0 / self.fps,
            "joint_order": list(POLICY_JOINTS),
            "frames": self.frames,
            "events": self.events,
            "summary": summary,
        }
