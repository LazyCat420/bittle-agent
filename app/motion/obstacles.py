"""Obstacle Course & Terrain Geometry Engine for Petoi Bittle.

Defines 3D obstacle courses, geometric boundaries, and kinematic terrain
clearance verification for stair climbing, ramp balancing, tunnel crawling,
and agility slalom navigation.
"""

from __future__ import annotations

from typing import Any

COURSE_PRESETS = {
    "none": "Flat ground arena",
    "mini_stairs": "3-step modular staircase (18mm risers, 80mm treads, 240mm width)",
    "ramp_bridge": "18° incline ramp (220mm), elevated balance beam (40mm high, 120mm wide), decline ramp",
    "crawl_tunnel": "Low clearance crawl tunnel (65mm ceiling, 160mm width, 280mm length)",
    "agility_slalom": "4-cone precision slalom course (spaced 100mm apart, zigzag offset ±45mm)",
}

COURSE_LAYOUTS: dict[str, dict[str, Any]] = {
    "none": {
        "preset": "none",
        "description": "Flat ground arena",
        "obstacles": [],
        "max_elevation_m": 0.0,
        "min_clearance_m": 1.0,
    },
    "mini_stairs": {
        "preset": "mini_stairs",
        "description": "3-step modular staircase scaled to Bittle's 88mm standing height",
        "step_count": 3,
        "riser_height_m": 0.018,  # 18mm
        "tread_depth_m": 0.080,   # 80mm
        "step_width_m": 0.240,    # 240mm (across Z)
        "start_x_m": 0.140,       # 140mm forward along X
        "start_z_m": 0.140,       # alias for compatibility
        "landing_depth_m": 0.140,
        "total_elevation_m": 0.054, # 54mm
        "obstacles": [
            {"type": "step", "index": 1, "elevation_m": 0.018, "min_x": 0.140, "max_x": 0.220, "min_z": 0.140, "max_z": 0.220},
            {"type": "step", "index": 2, "elevation_m": 0.036, "min_x": 0.220, "max_x": 0.300, "min_z": 0.220, "max_z": 0.300},
            {"type": "step", "index": 3, "elevation_m": 0.054, "min_x": 0.300, "max_x": 0.380, "min_z": 0.300, "max_z": 0.380},
            {"type": "landing", "index": 4, "elevation_m": 0.054, "min_x": 0.380, "max_x": 0.520, "min_z": 0.380, "max_z": 0.520},
        ],
        "constraints": {
            "required_foot_lift_min_mm": 18.0,
            "max_tolerable_pitch_deg": 35.0,
        }
    },
    "ramp_bridge": {
        "preset": "ramp_bridge",
        "description": "Ascending 10.3° ramp to elevated narrow beam and descending ramp",
        "ramp_length_m": 0.220,
        "ramp_height_m": 0.040,
        "ramp_width_m": 0.140,
        "bridge_length_m": 0.200,
        "start_x_m": 0.130,
        "start_z_m": 0.130,
        "obstacles": [
            {"type": "up_ramp", "start_x": 0.130, "end_x": 0.350, "start_z": 0.130, "end_z": 0.350, "start_y": 0.0, "end_y": 0.040, "angle_deg": 10.3},
            {"type": "bridge", "start_x": 0.350, "end_x": 0.550, "start_z": 0.350, "end_z": 0.550, "elevation_y": 0.040, "width": 0.140},
            {"type": "down_ramp", "start_x": 0.550, "end_x": 0.770, "start_z": 0.550, "end_z": 0.770, "start_y": 0.040, "end_y": 0.0, "angle_deg": -10.3},
        ],
        "constraints": {
            "max_lateral_drift_mm": 50.0,
            "pitch_compensation_deg": 10.0,
        }
    },
    "crawl_tunnel": {
        "preset": "crawl_tunnel",
        "description": "Low clearance crawl tunnel requiring crouch or belly-crawl posture",
        "tunnel_length_m": 0.280,
        "tunnel_width_m": 0.160,
        "ceiling_height_m": 0.065, # 65mm ceiling
        "start_x_m": 0.120,
        "start_z_m": 0.120,
        "obstacles": [
            {
                "type": "tunnel",
                "start_x": 0.120,
                "end_x": 0.400,
                "start_z": 0.120,
                "end_z": 0.400,
                "ceiling_height_m": 0.065,
                "clearance_width_m": 0.160,
            }
        ],
        "constraints": {
            "max_body_height_mm": 62.0,
            "required_gait": "crawl_or_crouch",
        }
    },
    "agility_slalom": {
        "preset": "agility_slalom",
        "description": "Zigzag precision slalom course around 4 traffic cones",
        "cone_count": 4,
        "cone_radius_m": 0.022,
        "cone_height_m": 0.055,
        "start_x_m": 0.140,
        "obstacles": [
            {"type": "cone", "index": 1, "x": 0.140, "z": 0.045, "radius_m": 0.022},
            {"type": "cone", "index": 2, "x": 0.240, "z": -0.045, "radius_m": 0.022},
            {"type": "cone", "index": 3, "x": 0.340, "z": 0.045, "radius_m": 0.022},
            {"type": "cone", "index": 4, "x": 0.440, "z": -0.045, "radius_m": 0.022},
        ],
        "constraints": {
            "min_turn_radius_m": 0.10,
            "clearance_margin_mm": 15.0,
        }
    }
}


def get_course_layout(preset_name: str) -> dict[str, Any]:
    """Retrieve verified obstacle layout definition."""
    return COURSE_LAYOUTS.get(preset_name, COURSE_LAYOUTS["none"])


def evaluate_terrain_clearance(
    course_preset: str,
    angles_or_sequence: dict[str | int, float] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Kinematically evaluate if joint angles or sequence clear the chosen obstacle course.
    
    Checks:
    - Mini stairs: verifies front knee flexion (joint 12, 13) achieves >= 18mm vertical foot lift.
    - Crawl tunnel: verifies shoulder and knee extension keeps torso + head <= 65mm ceiling.
    - Ramp bridge: verifies body pitch stability and stance width within beam limits.
    - Slalom: checks steering yaw angle capability.
    """
    layout = get_course_layout(course_preset)
    if course_preset == "none":
        return {
            "ok": True,
            "course": "none",
            "clears": True,
            "margin_mm": 100.0,
            "diagnostics": "Flat ground arena has no terrain obstructions.",
            "issues": [],
            "recommendations": [],
        }

    # Normalize input into a list of angle frames
    frames: list[dict[int, float]] = []
    if isinstance(angles_or_sequence, dict):
        # Single frame
        angles_map = {int(k): float(v) for k, v in angles_or_sequence.items() if str(k).isdigit()}
        frames.append(angles_map)
    elif isinstance(angles_or_sequence, list):
        for item in angles_or_sequence:
            if isinstance(item, dict):
                sub_angles = item.get("angles", item)
                angles_map = {int(k): float(v) for k, v in sub_angles.items() if str(k).isdigit()}
                if angles_map:
                    frames.append(angles_map)

    if not frames:
        return {
            "ok": False,
            "course": course_preset,
            "clears": False,
            "error": "No valid joint angle frames provided for evaluation.",
            "issues": ["Empty or invalid frame data"],
            "recommendations": ["Provide angles dictionary or sequence steps"],
        }

    issues: list[str] = []
    recommendations: list[str] = []
    clears = True
    min_margin_mm = 999.0

    # 1. Mini Stairs evaluation
    if course_preset == "mini_stairs":
        # Required: front paws must be lifted higher than the 18mm step riser during the stride.
        # On Bittle, neutral stand is knee=80, shoulder=-45.
        # To lift the foot by >=18mm, front knees (12, 13) must flex to >= 92° or shoulders retract to >= -30°.
        max_fl_knee = max(f.get(12, 80) for f in frames)
        max_fr_knee = max(f.get(13, 80) for f in frames)
        max_knee_lift = max(max_fl_knee, max_fr_knee)

        # Foot lift estimate: ~0.85mm per degree of knee flexion past 80°
        estimated_lift_mm = max(0.0, (max_knee_lift - 80.0) * 0.85)
        riser_mm = layout["riser_height_m"] * 1000.0
        margin_mm = estimated_lift_mm - riser_mm

        if margin_mm < 0:
            clears = False
            issues.append(
                f"Insufficient foot lift: peak knee flexion {max_knee_lift:.1f}° achieves only ~{estimated_lift_mm:.1f}mm lift, which trips on the {riser_mm:.1f}mm step riser."
            )
            recommendations.append(
                "Increase knee flexion on joints 12 & 13 to at least 95° to clear the 18mm step riser."
            )
            min_margin_mm = margin_mm
        else:
            min_margin_mm = min(min_margin_mm, margin_mm)

    # 2. Crawl Tunnel evaluation
    elif course_preset == "crawl_tunnel":
        # Ceiling is 65mm. Neutral standing height is 88mm.
        # Belly crawl (like `rest` or `crF`) holds knees around 40-60° and shoulders around -55° or 60°,
        # reducing body height to ~45-50mm.
        ceiling_mm = layout["ceiling_height_m"] * 1000.0

        for idx, frame in enumerate(frames):
            # Estimate body height from front shoulders (8, 9) and knees (12, 13)
            sh = (abs(frame.get(8, -45)) + abs(frame.get(9, -45))) / 2.0
            kn = (frame.get(12, 80) + frame.get(13, 80)) / 2.0

            # Kinematic height approximation for Bittle
            # Standing (sh=-45, kn=80) -> ~88mm (exceeds 65mm tunnel ceiling).
            # Belly crawl / crouch (kn <= 60, splayed limbs) -> ~46-52mm (clears 65mm ceiling with headroom).
            r_kn = (frame.get(14, 80) + frame.get(15, 80)) / 2.0
            sh_f = abs(frame.get(8, -45))
            sh_r = abs(frame.get(10, -45))
            if kn <= 60 and r_kn <= 60 and (sh_f >= 48 or sh_r >= 48):
                # Belly crawl or low rest posture
                est_height_mm = 48.0
            else:
                # Upright or partially standing stance
                leg_lift = (max(0.0, kn - 40.0) / 40.0) * 46.0
                est_height_mm = 42.0 + min(46.0, leg_lift)

            margin_mm = ceiling_mm - est_height_mm
            if margin_mm < 0:
                clears = False
                issues.append(
                    f"Frame {idx}: Estimated height {est_height_mm:.1f}mm exceeds tunnel ceiling {ceiling_mm:.1f}mm."
                )
                recommendations.append(
                    "Transition to low crawl posture (knee angles <= 55°, belly close to ground) to enter tunnel."
                )
            min_margin_mm = min(min_margin_mm, margin_mm)

    # 3. Ramp Bridge evaluation
    elif course_preset == "ramp_bridge":
        # Check torso pitch: rear knees vs front knees to avoid flipping backwards on 18° incline
        for idx, frame in enumerate(frames):
            f_kn = (frame.get(12, 80) + frame.get(13, 80)) / 2.0
            r_kn = (frame.get(14, 80) + frame.get(15, 80)) / 2.0
            # If front legs are much taller than rear legs, center of mass pitches backwards
            pitch_bias = f_kn - r_kn
            if pitch_bias > 30.0:
                clears = False
                issues.append(
                    f"Frame {idx}: Extreme backwards pitch bias ({pitch_bias:.1f}° differential) risks tipping over backward on the incline ramp."
                )
                recommendations.append(
                    "Compress front knees and extend rear legs slightly to maintain forward center-of-mass on the ramp."
                )
        min_margin_mm = min(min_margin_mm, 20.0 if clears else -10.0)

    # 4. Agility Slalom evaluation
    elif course_preset == "agility_slalom":
        # Check if sequence uses head pan (joint 0) or differential steering to navigate cones
        has_steering = any(abs(f.get(0, 0)) >= 15 or abs(f.get(8, -45) - f.get(9, -45)) >= 10 for f in frames)
        if not has_steering and len(frames) > 2:
            issues.append("Sequence walks straight ahead with zero steering or head scan; will collide with cone 1.")
            recommendations.append("Incorporate lateral yaw turns (joint 0 pan + differential leg angles) to weave through cones.")
            clears = False
            min_margin_mm = -15.0
        else:
            min_margin_mm = 25.0

    return {
        "ok": True,
        "course": course_preset,
        "clears": clears,
        "margin_mm": round(min_margin_mm, 1),
        "issues": issues,
        "recommendations": recommendations,
        "frame_count": len(frames),
    }


def get_obstacle_proximity(course_preset: str, current_x: float, current_z: float = 0.0) -> dict[str, Any]:
    """Calculate distance and metadata for the next terrain obstacle along the forward path."""
    if course_preset == "mini_stairs":
        steps = [
            {"type": "step", "index": 1, "riser_x": 0.140, "tread_end_x": 0.220, "elevation_m": 0.018},
            {"type": "step", "index": 2, "riser_x": 0.220, "tread_end_x": 0.300, "elevation_m": 0.036},
            {"type": "step", "index": 3, "riser_x": 0.300, "tread_end_x": 0.380, "elevation_m": 0.054},
            {"type": "landing", "index": 4, "riser_x": 0.380, "tread_end_x": 0.520, "elevation_m": 0.054},
        ]
        if current_x < 0.140:
            return {
                "active_course": "mini_stairs",
                "current_zone": "approach",
                "next_obstacle": "step_1",
                "distance_to_riser_m": round(0.140 - current_x, 3),
                "riser_height_mm": 18.0,
                "current_elevation_mm": 0.0,
            }
        for s in steps[:3]:
            if s["riser_x"] <= current_x < s["tread_end_x"]:
                next_step = steps[s["index"]]
                return {
                    "active_course": "mini_stairs",
                    "current_zone": f"step_{s['index']}",
                    "current_elevation_mm": round(s["elevation_m"] * 1000, 1),
                    "next_obstacle": f"{next_step['type']}_{next_step['index']}",
                    "distance_to_riser_m": round(next_step["riser_x"] - current_x, 3),
                    "riser_height_mm": 18.0 if next_step["type"] == "step" else 0.0,
                }
        if current_x >= 0.380:
            return {
                "active_course": "mini_stairs",
                "current_zone": "top_landing_platform",
                "next_obstacle": "none_goal_reached",
                "distance_to_riser_m": 0.0,
                "riser_height_mm": 0.0,
                "current_elevation_mm": 54.0,
            }

    return {
        "active_course": course_preset,
        "current_zone": "open_ground",
        "next_obstacle": "none",
        "distance_to_riser_m": 99.0,
        "riser_height_mm": 0.0,
        "current_elevation_mm": 0.0,
    }


def simulate_stair_climb_episode(
    angles_or_sequence: dict[str | int, float] | list[dict[str, Any]],
    adjustments: dict[str, float] | None = None,
    start_x: float = 0.05,
) -> dict[str, Any]:
    """Simulate an autonomous training episode for Bittle climbing the 3-step 18mm staircase.
    
    Evaluates kinematic trajectory, per-stride forward displacement, knee clearance,
    body pitch stability, and contact outcome across all 3 steps.
    """
    import math

    adjustments = adjustments or {}
    knee_adjust = float(adjustments.get("knee_lift_deg", 0.0))
    pitch_adjust = float(adjustments.get("pitch_compensation_deg", 0.0))
    stride_mult = float(adjustments.get("stride_mult", 1.0))

    frames: list[dict[int, float]] = []
    if isinstance(angles_or_sequence, dict):
        angles_map = {int(k): float(v) for k, v in angles_or_sequence.items() if str(k).isdigit()}
        frames.append(angles_map)
    elif isinstance(angles_or_sequence, list):
        for item in angles_or_sequence:
            sub = item.get("angles", item) if isinstance(item, dict) else {}
            angles_map = {int(k): float(v) for k, v in sub.items() if str(k).isdigit()}
            if angles_map:
                frames.append(angles_map)

    if not frames:
        return {
            "success": False,
            "outcome": "invalid_input",
            "error": "No valid joint angle frames provided for simulation.",
        }

    # Evaluate peak knee lift with adjustment
    max_fl_knee = max(f.get(12, 80) for f in frames) + knee_adjust
    max_fr_knee = max(f.get(13, 80) for f in frames) + knee_adjust
    peak_knee = max(max_fl_knee, max_fr_knee)
    foot_lift_mm = max(0.0, (peak_knee - 80.0) * 0.85)

    wheelbase = 0.120  # 120mm
    riser_height_m = 0.018  # 18mm
    riser_height_mm = 18.0

    steps = [
        {"idx": 1, "riser_x": 0.140, "tread_end_x": 0.220, "elevation": 0.018},
        {"idx": 2, "riser_x": 0.220, "tread_end_x": 0.300, "elevation": 0.036},
        {"idx": 3, "riser_x": 0.300, "tread_end_x": 0.380, "elevation": 0.054},
    ]

    current_x = start_x
    stride_len = 0.038 * stride_mult
    stride_count = 0
    max_strides = 16
    telemetry: list[dict[str, Any]] = []
    current_front_elevation = 0.0
    current_rear_elevation = 0.0
    max_pitch_deg = 0.0

    while current_x < 0.400 and stride_count < max_strides:
        stride_count += 1
        current_x += stride_len
        front_x = current_x + 0.050
        rear_x = current_x - 0.070

        # Check front paw step riser collisions
        for s in steps:
            # If front paw has just reached or crossed this riser
            if front_x >= s["riser_x"] and current_front_elevation < s["elevation"]:
                if foot_lift_mm < riser_height_mm:
                    deficit = riser_height_mm - foot_lift_mm
                    telemetry.append({
                        "stride": stride_count,
                        "event": "trip",
                        "step_index": s["idx"],
                        "riser_x": s["riser_x"],
                        "foot_lift_mm": round(foot_lift_mm, 1),
                        "required_mm": riser_height_mm,
                    })
                    return {
                        "success": False,
                        "outcome": "tripped",
                        "failed_at_step": s["idx"],
                        "failed_at_x": round(s["riser_x"], 3),
                        "strides_completed": stride_count,
                        "peak_foot_lift_mm": round(foot_lift_mm, 1),
                        "required_clearance_mm": riser_height_mm,
                        "deficit_mm": round(deficit, 1),
                        "max_pitch_deg": round(max_pitch_deg, 1),
                        "telemetry_log": telemetry,
                        "reflection": (
                            f"EPISODE FAILED: Front paws tripped on Step {s['idx']} riser (x={s['riser_x']}m). "
                            f"Achieved foot lift was {foot_lift_mm:.1f}mm ({peak_knee:.1f}° knee flexion), "
                            f"which is {deficit:.1f}mm below the required {riser_height_mm}mm riser height."
                        ),
                        "guidance": (
                            f"Increase front knee flexion (joints 12 & 13) by at least +{math.ceil(deficit / 0.85)}° "
                            f"(suggested >= {peak_knee + deficit / 0.85 + 2:.0f}°) to clear the 18mm riser."
                        ),
                    }
                else:
                    current_front_elevation = s["elevation"]
                    telemetry.append({
                        "stride": stride_count,
                        "event": "step_cleared",
                        "step_index": s["idx"],
                        "elevation_mm": round(s["elevation"] * 1000, 1),
                    })

            # Check rear paw step riser transitions
            if rear_x >= s["riser_x"] and current_rear_elevation < s["elevation"]:
                current_rear_elevation = s["elevation"]

        # Calculate pitch
        pitch_rad = math.atan2(current_front_elevation - current_rear_elevation, wheelbase)
        pitch_deg = math.degrees(pitch_rad) + pitch_adjust
        max_pitch_deg = max(max_pitch_deg, abs(pitch_deg))

        if pitch_deg > 35.0:
            return {
                "success": False,
                "outcome": "backward_tip_over",
                "failed_at_step": s["idx"],
                "pitch_deg": round(pitch_deg, 1),
                "strides_completed": stride_count,
                "peak_foot_lift_mm": round(foot_lift_mm, 1),
                "telemetry_log": telemetry,
                "reflection": f"EPISODE FAILED: Excessive backward pitch ({pitch_deg:.1f}° > 35°) caused robot to tip over backward.",
                "guidance": "Extend rear legs and apply forward torso pitch bias to stabilize center-of-mass.",
            }

    # Final landing platform reached check
    if current_x >= 0.380:
        stability = max(50.0, 100.0 - (max_pitch_deg * 1.2))
        return {
            "success": True,
            "outcome": "climbed_stairs_successfully",
            "final_x": round(current_x, 3),
            "final_elevation_mm": 54.0,
            "strides_completed": stride_count,
            "peak_foot_lift_mm": round(foot_lift_mm, 1),
            "max_pitch_deg": round(max_pitch_deg, 1),
            "stability_score": round(stability, 1),
            "steps_cleared": 3,
            "telemetry_log": telemetry,
            "reflection": (
                f"EPISODE SUCCESS: Cleared all 3 steps onto top landing platform in {stride_count} strides. "
                f"Foot lift: {foot_lift_mm:.1f}mm (margin: +{foot_lift_mm - riser_height_mm:.1f}mm). "
                f"Peak body pitch: {max_pitch_deg:.1f}°. Stability score: {stability:.1f}%."
            ),
        }

    return {
        "success": False,
        "outcome": "stalled",
        "final_x": round(current_x, 3),
        "strides_completed": stride_count,
        "peak_foot_lift_mm": round(foot_lift_mm, 1),
        "telemetry_log": telemetry,
        "reflection": f"Episode terminated after {stride_count} strides before reaching landing platform (reached x={current_x:.3f}m).",
        "guidance": "Increase stride length multiplier or number of steps.",
    }


def compute_leg_reach(shoulder_deg: float, knee_deg: float) -> float:
    """Compute vertical drop (in meters) from torso shoulder pivot to foot tip."""
    import math

    def rot_y(v, deg):
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)

    def add_v(v1, v2):
        return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

    v_thigh = (-0.044433, 0.00142, -0.011906)
    v_shank = (0.047813, 0.013, -0.009684)

    v_thigh_rot = rot_y(v_thigh, shoulder_deg)
    p_knee = add_v((0.0525, -0.0485, 0.022), v_thigh_rot)
    v_shank_rot = rot_y(rot_y(v_shank, knee_deg), shoulder_deg)
    p_foot = add_v(p_knee, v_shank_rot)

    # In CAD coords, +Z is up and ground is -Z. Vertical drop from torso center is -p_foot[2]
    return -p_foot[2]


def compute_posture_ground_contact(angles: dict[int | str, float]) -> dict[str, Any]:
    """Compute ground contact clearances, chassis height, and pitch under gravity.
    
    In standing: h_front ~ 53.2mm, h_rear ~ 53.2mm -> height = 53.2mm, pitch = 0 deg
    In sitting: h_front ~ 48.4mm, rear folded up -> rump rests on floor at 22mm -> height ~ 35mm, pitch ~ +12.4 deg
    In resting: legs folded -> belly on floor at 20mm -> height = 20mm, pitch = 0 deg
    """
    import math

    normalized = {int(k): float(v) for k, v in angles.items() if str(k).isdigit()}

    # Front legs (8=FL sh, 12=FL kn, 9=FR sh, 13=FR kn)
    fl_reach = compute_leg_reach(normalized.get(8, -45.0), normalized.get(12, 80.0))
    fr_reach = compute_leg_reach(normalized.get(9, -45.0), normalized.get(13, 80.0))
    # Chest minimum contact floor: 20mm (bottom of battery / chassis)
    h_front = max(0.020, (fl_reach + fr_reach) / 2.0)

    # Rear legs (11=BL sh, 15=BL kn, 10=BR sh, 14=BR kn)
    bl_reach = compute_leg_reach(normalized.get(11, -45.0), normalized.get(15, 80.0))
    br_reach = compute_leg_reach(normalized.get(10, -45.0), normalized.get(14, 80.0))
    # Rump minimum contact floor: 22mm (bottom of pelvis / rear battery)
    h_rear = max(0.022, (bl_reach + br_reach) / 2.0)

    wheelbase = 0.120  # 120mm
    chassis_height = (h_front + h_rear) / 2.0
    pitch_rad = math.atan2(h_front - h_rear, wheelbase)
    pitch_deg = math.degrees(pitch_rad)

    return {
        "h_front_m": round(h_front, 4),
        "h_rear_m": round(h_rear, 4),
        "chassis_height_m": round(chassis_height, 4),
        "chassis_height_mm": round(chassis_height * 1000.0, 1),
        "pitch_deg": round(pitch_deg, 1),
        "is_sitting": bool(h_rear <= 0.025 and normalized.get(10, 0) >= 70 and normalized.get(11, 0) >= 70),
        "is_resting": bool(chassis_height <= 0.032 and normalized.get(10, 0) < 65),
        "is_standing": bool(chassis_height >= 0.048),
    }

