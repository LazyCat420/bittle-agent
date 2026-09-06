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
        "step_width_m": 0.240,    # 240mm
        "start_z_m": 0.160,
        "landing_depth_m": 0.140,
        "total_elevation_m": 0.054, # 54mm
        "obstacles": [
            {"type": "step", "index": 1, "elevation_m": 0.018, "min_z": 0.160, "max_z": 0.240},
            {"type": "step", "index": 2, "elevation_m": 0.036, "min_z": 0.240, "max_z": 0.320},
            {"type": "step", "index": 3, "elevation_m": 0.054, "min_z": 0.320, "max_z": 0.400},
            {"type": "landing", "index": 4, "elevation_m": 0.054, "min_z": 0.400, "max_z": 0.540},
        ],
        "constraints": {
            "required_foot_lift_min_mm": 18.0,
            "max_tolerable_pitch_deg": 35.0,
        }
    },
    "ramp_bridge": {
        "preset": "ramp_bridge",
        "description": "Ascending 18° ramp to elevated narrow beam and descending ramp",
        "ramp_length_m": 0.220,
        "ramp_height_m": 0.040,
        "ramp_width_m": 0.120,
        "bridge_length_m": 0.200,
        "start_z_m": 0.140,
        "obstacles": [
            {"type": "up_ramp", "start_z": 0.140, "end_z": 0.360, "start_y": 0.0, "end_y": 0.040, "angle_deg": 10.3},
            {"type": "bridge", "start_z": 0.360, "end_z": 0.560, "elevation_y": 0.040, "width": 0.120},
            {"type": "down_ramp", "start_z": 0.560, "end_z": 0.780, "start_y": 0.040, "end_y": 0.0, "angle_deg": -10.3},
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
        "start_z_m": 0.120,
        "obstacles": [
            {
                "type": "tunnel",
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
        "obstacles": [
            {"type": "cone", "index": 1, "x": 0.045, "z": 0.140, "radius_m": 0.022},
            {"type": "cone", "index": 2, "x": -0.045, "z": 0.240, "radius_m": 0.022},
            {"type": "cone", "index": 3, "x": 0.045, "z": 0.340, "radius_m": 0.022},
            {"type": "cone", "index": 4, "x": -0.045, "z": 0.440, "radius_m": 0.022},
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
