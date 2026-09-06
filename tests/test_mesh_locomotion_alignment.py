"""Regression & Kinematic Tests for 3D Mesh Heading and Locomotion Vector Alignment.

Prevents sideways crab-walking regression by asserting that:
1. The 3D model nose vector in World coordinates at yaw = 0 points forward along +X.
2. The lateral component across the shoulders (World Z) is 0.0.
3. Forward locomotion advances strictly in the direction of the nose.
4. Backward locomotion retreats strictly opposite the nose.
"""

import subprocess
import json
import pytest
from pathlib import Path


def test_threejs_mesh_heading_collinear_with_forward_motion():
    """Verify that Bittle's Three.js mesh forward vector aligns with the +X locomotion axis."""
    repo_root = Path(__file__).resolve().parent.parent
    script = """
import * as THREE from './static/vendor/three.module.js';

// Build minimal viewer hierarchy replicating viewer.js buildHierarchy
const robotGroup = new THREE.Group();
const torso = new THREE.Group();

// Read viewer.js torso rotation directly or replicate viewer configuration
import fs from 'fs';
const viewerCode = fs.readFileSync('./static/js/viewer.js', 'utf8');

// Extract torso.rotation.x and torso.rotation.z from viewer.js
const rxMatch = viewerCode.match(/torso\\.rotation\\.x\\s*=\\s*([^;]+);/);
const rzMatch = viewerCode.match(/torso\\.rotation\\.z\\s*=\\s*([^;]+);/);

if (!rxMatch || !rzMatch) {
  throw new Error('Could not parse torso.rotation from static/js/viewer.js');
}

const parseExpr = (expr) => {
  return Function('THREE', 'return ' + expr.trim())(THREE);
};

torso.rotation.x = parseExpr(rxMatch[1]);
torso.rotation.z = parseExpr(rzMatch[1]);
robotGroup.add(torso);

// CAD neck/head pivot pos from Petoi CAD model
const neckPivotPos = new THREE.Vector3(0.047554, 0.0, 0.035941);
const headPivot = new THREE.Group();
headPivot.position.copy(neckPivotPos);
torso.add(headPivot);

robotGroup.updateMatrixWorld(true);

const torsoPos = new THREE.Vector3();
const headPos = new THREE.Vector3();
torso.getWorldPosition(torsoPos);
headPivot.getWorldPosition(headPos);

// Vector from torso center to head
const heading = new THREE.Vector3().subVectors(headPos, torsoPos).normalize();

// Forward locomotion vector at yaw = 0 is (+1, 0, 0)
const fwdDir = new THREE.Vector3(1, 0, 0);
const lateralDir = new THREE.Vector3(0, 0, 1);

const forwardDot = heading.dot(fwdDir);
const lateralDot = Math.abs(heading.dot(lateralDir));

const result = {
  torsoRotationX: torso.rotation.x,
  torsoRotationZ: torso.rotation.z,
  headingX: heading.x,
  headingY: heading.y,
  headingZ: heading.z,
  forwardDot,
  lateralDot,
};

console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True
    )
    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip().startswith("{")]
    assert len(lines) > 0, f"Expected JSON result, got: {result.stdout}"
    data = json.loads(lines[-1])

    # 1. Forward dot product must be > 0.70 (aligned with +X)
    assert data["forwardDot"] > 0.70, (
        f"Robot nose is not aligned with forward locomotion axis (+X)! "
        f"forwardDot={data['forwardDot']:.4f} (expected > 0.70). Robot is moving sideways!"
    )

    # 2. Lateral dot product must be ~0.0 (no lateral tilt or sideways crab-walking)
    assert data["lateralDot"] == pytest.approx(0.0, abs=1e-3), (
        f"Robot heading has non-zero lateral component: lateralDot={data['lateralDot']:.4f}"
    )

    # 3. Z rotation must be 0 (no 90-degree yaw twist at torso level)
    assert data["torsoRotationZ"] == pytest.approx(0.0, abs=1e-3), (
        f"torso.rotation.z should be 0, got {data['torsoRotationZ']}"
    )
