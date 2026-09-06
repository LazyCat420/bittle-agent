"""Tests for Bittle Locomotion Speed and Gait Direction Resolver.

Validates that getLocomotionSpeed accurately maps forward gaits, backward gaits,
turns, and obstacle gaits to non-zero physical displacement vectors in 3D space.
"""

import json
import subprocess
from pathlib import Path
import pytest

VIEWER_JS_PATH = Path(__file__).resolve().parent.parent / "static" / "js" / "viewer.js"


def evaluate_js_locomotion_speed(sequence_names: list[str]) -> dict[str, dict[str, float]]:
    """Evaluates getLocomotionSpeed directly in Node.js using viewer.js."""
    script = f"""
    import fs from 'fs';
    const content = fs.readFileSync('{VIEWER_JS_PATH}', 'utf8');

    // Extract getLocomotionSpeed method body between start and next method
    const startStr = 'getLocomotionSpeed(sequenceName) {{';
    const endStr = 'driveForward(loop = true) {{';
    const start = content.indexOf(startStr);
    const end = content.indexOf(endStr);
    if (start === -1 || end === -1) {{
        console.error("Failed to locate getLocomotionSpeed in viewer.js");
        process.exit(1);
    }}

    const methodBody = content.slice(start + startStr.length, end).trim();
    const cleanedBody = methodBody.replace(/\\}}$/, '').trim();
    const fn = new Function('sequenceName', cleanedBody);

    const names = {json.dumps(sequence_names)};
    const results = {{}};
    for (const name of names) {{
        results[name] = fn(name);
    }}
    console.log(JSON.stringify(results));
    """
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(proc.stdout)


def test_forward_locomotion_speeds():
    """Forward tokens, labels, and aliases must produce positive forward velocity vx > 0."""
    queries = [
        "wkF",
        "Walk forward",
        "walk_forward",
        "walk",
        "forward",
        "step forward",
        "trot",
        "trF",
        "crawl",
        "crF",
        "run",
        "march",
        "stalk",
        "stair step",
        "ramp climb"
    ]
    results = evaluate_js_locomotion_speed(queries)
    for q in queries:
        speed = results[q]
        assert speed["vx"] > 0, f"Expected positive forward speed for '{q}', got {speed}"
        if "stair" in q or "ramp" in q:
            assert speed["vyaw"] == 0, f"Obstacle climb should be straight forward, got {speed}"


def test_backward_locomotion_speeds():
    """Backward tokens, labels, and aliases must produce negative forward velocity vx < 0."""
    queries = [
        "bk",
        "Back up",
        "backup",
        "back",
        "backward",
        "reverse",
        "retreat",
        "back up"
    ]
    results = evaluate_js_locomotion_speed(queries)
    for q in queries:
        speed = results[q]
        assert speed["vx"] < 0, f"Expected negative backward speed for '{q}', got {speed}"
        assert speed["vyaw"] == 0, f"Expected zero yaw rate for straight reverse '{q}', got {speed}"


def test_turning_locomotion_speeds():
    """Turn gaits must produce non-zero yaw velocities (left > 0, right < 0)."""
    left_queries = ["wkL", "Walk left", "turn left", "step left", "left"]
    right_queries = ["wkR", "Walk right", "turn right", "step right", "right"]

    all_queries = left_queries + right_queries
    results = evaluate_js_locomotion_speed(all_queries)

    for q in left_queries:
        speed = results[q]
        assert speed["vx"] > 0, f"Turning left should still have forward drive, got {speed}"
        assert speed["vyaw"] > 0, f"Left turn must have positive yaw rate (counter-clockwise), got {speed}"

    for q in right_queries:
        speed = results[q]
        assert speed["vx"] > 0, f"Turning right should still have forward drive, got {speed}"
        assert speed["vyaw"] < 0, f"Right turn must have negative yaw rate (clockwise), got {speed}"


def test_stationary_postures_produce_zero_speed():
    """Resting and balanced postures must return exactly 0 velocity."""
    queries = ["sit", "balance", "stand", "rest", "zero", "calib", "unknown_xyz"]
    results = evaluate_js_locomotion_speed(queries)
    for q in queries:
        speed = results[q]
        assert speed["vx"] == 0.0, f"Posture '{q}' should have vx = 0, got {speed}"
        assert speed["vyaw"] == 0.0, f"Posture '{q}' should have vyaw = 0, got {speed}"
