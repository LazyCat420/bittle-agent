"""OpenCat Skill Compiler.

Compiles validated Skill IR into OpenCat binary skill data format and produces
an immutable SkillManifest cryptographically signed with SHA-256.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import struct
from typing import Any

from ..joints import DOF, STAND_POSE
from ..profiles import HardwareProfile, get_registry
from .schema import SkillIR, SkillManifest
from .validator import TrajectoryValidator


class SkillCompilerError(Exception):
    pass


class SkillCompiler:
    COMPILER_VERSION = "1.0.0"

    def __init__(self, profile: HardwareProfile | None = None):
        self.profile = profile
        self.validator = TrajectoryValidator(profile)

    def compile(self, ir: SkillIR) -> SkillManifest:
        # 1. Deterministic Validation Gate
        val_res = self.validator.validate(ir)
        if not val_res.valid:
            raise SkillCompilerError(f"Cannot compile invalid Skill IR: {'; '.join(val_res.errors)}")

        profile = self.profile
        if profile is None or profile.profile_id != ir.profile_id:
            profile = get_registry().get(ir.profile_id)

        # 2. Canonical IR Hash
        ir_dict = ir.model_dump(mode="json")
        canonical_ir = json.dumps(ir_dict, sort_keys=True)
        ir_hash = hashlib.sha256(canonical_ir.encode("utf-8")).hexdigest()

        # 3. Assemble OpenCat Skill Array
        # Header:
        # - frame_count (period): signed char
        # - roll: signed char (0)
        # - pitch: signed char (0)
        # - angle_ratio: signed char (1)
        frame_count = len(ir.frames)
        header = [frame_count, 0, 0, 1]

        # Body:
        # For each frame: full 16 DOF angle array (defaulting to STAND_POSE for uncommanded joints)
        # followed by 4 control parameters for behaviors: speed, delay, trigger_axis, trigger_angle
        body: list[int] = []
        current_pose = {j: STAND_POSE.get(j, 0) for j in range(DOF)}

        for frame in ir.frames:
            # Update angles
            for joint_idx, angle in frame.angles_deg.items():
                current_pose[joint_idx] = int(round(angle))

            # Pack 16 joint angles in index order 0..15
            for j in range(DOF):
                body.append(current_pose.get(j, 0))

            # Pack behavior frame control metadata:
            # - speed: 1..127
            # - delay: 0..255 (in units of 50ms, clamped)
            # - trigger_axis: 0..3
            # - trigger_angle: signed char
            delay_units = min(255, max(0, int(round(frame.delay_ms / 50.0))))
            axis = frame.trigger_axis if frame.trigger_axis is not None else 0
            angle_thresh = frame.trigger_angle if frame.trigger_angle is not None else 0

            body.extend([
                int(frame.speed_deg_per_step),
                delay_units,
                int(axis),
                int(angle_thresh)
            ])

        raw_array = header + body

        # Encode for wire: Token 'K' + struct.pack('b' * len, *raw_array) + '~'
        # Check all values fit in signed char [-128, 127]
        packed_vals: list[int] = []
        for val in raw_array:
            clamped = max(-128, min(127, val))
            packed_vals.append(clamped)

        wire_payload = b"K" + struct.pack("b" * len(packed_vals), *packed_vals) + b"~"
        payload_hash = hashlib.sha256(wire_payload).hexdigest()

        manifest = SkillManifest(
            ir=ir,
            ir_hash=ir_hash,
            compiled_payload=wire_payload.hex(),
            payload_hash=payload_hash,
            compiler_version=self.COMPILER_VERSION,
            validation_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            simulated=False,
            approved_by=None,
            approved_at=None,
            promoted=False,
            status="validated",
        )

        return manifest
