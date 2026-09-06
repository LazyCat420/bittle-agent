"""Deterministic Trajectory and Safety Invariants Validator for Bittle Skill IR."""

from __future__ import annotations

from typing import Any

from ..joints import STAND_POSE, REST_POSE
from ..profiles import HardwareProfile, get_registry
from .schema import MotionBudget, SkillIR, ValidationResult

# Safety constants for trajectory admission
MAX_STEP_DELTA_DEG = 45.0          # Max degrees a joint may jump between adjacent frames
MAX_SKILL_TRAVEL_DEG = 720.0       # Max cumulative travel per joint in a single skill execution
MAX_DIRECTION_REVERSALS = 8        # Max rapid direction reversals (chatter/resonance prevention)
SAFE_SPEED_MAX = 20                # Speed values > 10 trigger warning; > 30 rejected for agent skills


class TrajectoryValidator:
    def __init__(self, profile: HardwareProfile | None = None):
        self.profile = profile

    def validate(self, ir: SkillIR) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Resolve Profile
        profile = self.profile
        if profile is None or profile.profile_id != ir.profile_id:
            try:
                profile = get_registry().get(ir.profile_id)
            except KeyError:
                return ValidationResult(
                    valid=False,
                    errors=[f"Hardware profile {ir.profile_id!r} not found in registry"],
                )

        # 2. Installed Joints Check
        for frame_idx, frame in enumerate(ir.frames):
            for joint_idx in frame.angles_deg:
                if not profile.is_installed(joint_idx):
                    errors.append(
                        f"Frame {frame_idx}: Joint {joint_idx} is not installed on hardware profile {profile.profile_id!r}"
                    )

        # 3. Agent Envelope Angle Limits
        for frame_idx, frame in enumerate(ir.frames):
            for joint_idx, angle in frame.angles_deg.items():
                if profile.is_installed(joint_idx):
                    env = profile.get_envelope(joint_idx)
                    if not (env.agent_min <= angle <= env.agent_max):
                        errors.append(
                            f"Frame {frame_idx}: Joint {joint_idx} angle {angle}° violates agent safety envelope [{env.agent_min}°, {env.agent_max}°]"
                        )

        # 4. Speed & Delay Checks
        for frame_idx, frame in enumerate(ir.frames):
            if frame.speed_deg_per_step > 30:
                errors.append(
                    f"Frame {frame_idx}: Speed {frame.speed_deg_per_step} deg/step exceeds safe agent limit (30)"
                )
            elif frame.speed_deg_per_step > 10:
                warnings.append(
                    f"Frame {frame_idx}: Speed {frame.speed_deg_per_step} deg/step is high (>10); ensure robot is stable"
                )

        # 5. Delta, Travel Budget & Direction Reversals Analysis
        total_travel: dict[int, float] = {j: 0.0 for j in profile.installed_joints}
        reversals: dict[int, int] = {j: 0 for j in profile.installed_joints}
        last_deltas: dict[int, float] = {j: 0.0 for j in profile.installed_joints}
        max_delta = 0.0

        # Establish baseline pose from entry_posture
        current_pose: dict[int, int] = {}
        if ir.entry_posture in ("balance", "stand", "up"):
            current_pose = {j: STAND_POSE.get(j, 0) for j in profile.installed_joints}
        elif ir.entry_posture == "rest":
            current_pose = {j: REST_POSE.get(j, 0) for j in profile.installed_joints}
        else:
            current_pose = {j: 0 for j in profile.installed_joints}

        # Check transition from entry_posture to Frame 0
        first_frame = ir.frames[0]
        for joint_idx, target_angle in first_frame.angles_deg.items():
            if joint_idx in current_pose:
                d = abs(target_angle - current_pose[joint_idx])
                if d > MAX_STEP_DELTA_DEG:
                    warnings.append(
                        f"Initial transition: Joint {joint_idx} jumps {d}° from entry posture '{ir.entry_posture}' to Frame 0"
                    )

        # Walk through frame sequence
        prev_angles = dict(current_pose)
        total_est_duration = 0.0

        for frame_idx, frame in enumerate(ir.frames):
            frame_max_delta = 0.0
            for joint_idx, angle in frame.angles_deg.items():
                if joint_idx not in prev_angles:
                    prev_angles[joint_idx] = angle
                    continue

                delta = angle - prev_angles[joint_idx]
                abs_delta = abs(delta)
                if abs_delta > max_delta:
                    max_delta = abs_delta
                if abs_delta > frame_max_delta:
                    frame_max_delta = abs_delta

                if abs_delta > MAX_STEP_DELTA_DEG:
                    errors.append(
                        f"Frame {frame_idx}: Joint {joint_idx} delta {abs_delta:.1f}° exceeds max step limit ({MAX_STEP_DELTA_DEG}°)"
                    )

                total_travel[joint_idx] = total_travel.get(joint_idx, 0.0) + abs_delta

                # Reversal check
                if abs_delta > 2.0:
                    last_d = last_deltas.get(joint_idx, 0.0)
                    if (delta > 0 and last_d < -2.0) or (delta < 0 and last_d > 2.0):
                        reversals[joint_idx] = reversals.get(joint_idx, 0) + 1
                    last_deltas[joint_idx] = delta

                prev_angles[joint_idx] = angle

            # Estimate duration for frame: time to sweep frame_max_delta at speed + delay
            step_speed = max(1, frame.speed_deg_per_step)
            # P1S servo roughly traverses ~320 deg/sec
            slew_time = frame_max_delta / (step_speed * 50.0)
            frame_duration = slew_time + (frame.delay_ms / 1000.0)
            total_est_duration += frame_duration

        # Multiplied by loop count
        total_est_duration *= ir.loop_count

        # Check total travel budget per joint
        max_travel_single = 0.0
        for joint_idx, travel in total_travel.items():
            scaled_travel = travel * ir.loop_count
            if scaled_travel > max_travel_single:
                max_travel_single = scaled_travel
            if scaled_travel > MAX_SKILL_TRAVEL_DEG:
                errors.append(
                    f"Joint {joint_idx} total travel {scaled_travel:.1f}° exceeds safety wear budget ({MAX_SKILL_TRAVEL_DEG}°)"
                )

        # Check reversal limits
        for joint_idx, rev_count in reversals.items():
            if rev_count * ir.loop_count > MAX_DIRECTION_REVERSALS:
                errors.append(
                    f"Joint {joint_idx} direction reversals ({rev_count * ir.loop_count}) exceed chatter limit ({MAX_DIRECTION_REVERSALS})"
                )

        budget = MotionBudget(
            max_delta_deg=round(max_delta, 1),
            total_travel_deg={j: round(t * ir.loop_count, 1) for j, t in total_travel.items() if t > 0},
            max_travel_single_joint_deg=round(max_travel_single, 1),
            direction_reversals={j: r * ir.loop_count for j, r in reversals.items() if r > 0},
            estimated_duration_sec=round(total_est_duration, 2),
            is_bounded=len(errors) == 0,
        )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            budget=budget,
        )
