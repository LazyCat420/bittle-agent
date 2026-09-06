/**
 * Bittle 3D Modular Obstacle Course & Terrain Engine
 * High-fidelity Three.js procedural obstacles for physical quadruped challenges.
 */

import * as THREE from 'three';

export const COURSE_PRESETS = {
  NONE: 'none',
  MINI_STAIRS: 'mini_stairs',
  RAMP_BRIDGE: 'ramp_bridge',
  CRAWL_TUNNEL: 'crawl_tunnel',
  AGILITY_SLALOM: 'agility_slalom'
};

export class ObstacleCourse {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'obstacle_course_group';
    this.scene.add(this.group);

    this.currentPreset = COURSE_PRESETS.NONE;
    this.colliders = []; // Array of bounding boxes or geometry descriptors
    this.materials = this.initMaterials();
  }

  initMaterials() {
    return {
      tread: new THREE.MeshStandardMaterial({
        color: 0x1c2128,
        roughness: 0.7,
        metalness: 0.2,
      }),
      riser: new THREE.MeshStandardMaterial({
        color: 0x0f141c,
        roughness: 0.5,
        metalness: 0.3,
      }),
      stripe: new THREE.MeshStandardMaterial({
        color: 0xffaa00,
        emissive: 0xff7700,
        emissiveIntensity: 0.4,
        roughness: 0.3,
        metalness: 0.1,
      }),
      ramp: new THREE.MeshStandardMaterial({
        color: 0x21262d,
        roughness: 0.6,
        metalness: 0.25,
      }),
      bridge: new THREE.MeshStandardMaterial({
        color: 0x161b22,
        roughness: 0.4,
        metalness: 0.5,
      }),
      tunnelArch: new THREE.MeshStandardMaterial({
        color: 0x30363d,
        roughness: 0.5,
        metalness: 0.4,
        side: THREE.DoubleSide,
      }),
      tunnelCeiling: new THREE.MeshStandardMaterial({
        color: 0x58a6ff,
        emissive: 0x1f6feb,
        emissiveIntensity: 0.25,
        roughness: 0.4,
        metalness: 0.2,
        transparent: true,
        opacity: 0.75,
      }),
      coneOrange: new THREE.MeshStandardMaterial({
        color: 0xff5500,
        roughness: 0.4,
        metalness: 0.1,
        emissive: 0xaa2200,
        emissiveIntensity: 0.2,
      }),
      coneWhite: new THREE.MeshStandardMaterial({
        color: 0xdddddd,
        roughness: 0.3,
        metalness: 0.1,
      }),
      waypoint: new THREE.MeshStandardMaterial({
        color: 0x3fb950,
        emissive: 0x2ea043,
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0.5,
      })
    };
  }

  clear() {
    while (this.group.children.length > 0) {
      const obj = this.group.children[0];
      this.group.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
    }
    this.colliders = [];
    this.currentPreset = COURSE_PRESETS.NONE;
  }

  loadPreset(presetName) {
    this.clear();
    this.currentPreset = presetName;

    switch (presetName) {
      case COURSE_PRESETS.MINI_STAIRS:
        this.buildMiniStairs();
        break;
      case COURSE_PRESETS.RAMP_BRIDGE:
        this.buildRampBridge();
        break;
      case COURSE_PRESETS.CRAWL_TUNNEL:
        this.buildCrawlTunnel();
        break;
      case COURSE_PRESETS.AGILITY_SLALOM:
        this.buildAgilitySlalom();
        break;
      case COURSE_PRESETS.NONE:
      default:
        // Plain flat arena
        break;
    }
  }

  /**
   * 1. Mini Stairs: 3 progressive steps scaled to Bittle's 88mm height
   * Each step: rise = 18mm (0.018m), tread = 80mm (0.08m), width = 240mm (0.24m)
   */
  buildMiniStairs() {
    const stepCount = 3;
    const riserHeight = 0.018; // 18mm - within reach of Bittle's 20-25mm foot lift
    const treadDepth = 0.080;  // 80mm
    const stepWidth = 0.240;   // 240mm
    const startZ = 0.160;      // 16cm in front of initial spawn

    for (let i = 0; i < stepCount; i++) {
      const currentElevation = (i + 1) * riserHeight;
      const centerZ = startZ + i * treadDepth + treadDepth / 2;
      const centerY = currentElevation / 2;

      // Solid block supporting the step
      const stepGeo = new THREE.BoxGeometry(stepWidth, currentElevation, treadDepth);
      const stepMesh = new THREE.Mesh(stepGeo, this.materials.tread);
      stepMesh.position.set(0, centerY, centerZ);
      stepMesh.castShadow = true;
      stepMesh.receiveShadow = true;
      this.group.add(stepMesh);

      // Warning edge stripe on each step nosing
      const stripeGeo = new THREE.BoxGeometry(stepWidth, 0.002, 0.006);
      const stripeMesh = new THREE.Mesh(stripeGeo, this.materials.stripe);
      stripeMesh.position.set(0, currentElevation + 0.001, centerZ + treadDepth / 2 - 0.003);
      this.group.add(stripeMesh);

      // Record collider info
      this.colliders.push({
        type: 'step',
        minX: -stepWidth / 2,
        maxX: stepWidth / 2,
        minZ: startZ + i * treadDepth,
        maxZ: startZ + (i + 1) * treadDepth,
        elevation: currentElevation,
        riser: riserHeight,
        index: i + 1,
      });
    }

    // Top landing platform
    const landingDepth = 0.140;
    const topElevation = stepCount * riserHeight;
    const landingZ = startZ + stepCount * treadDepth + landingDepth / 2;
    const landingGeo = new THREE.BoxGeometry(stepWidth, topElevation, landingDepth);
    const landingMesh = new THREE.Mesh(landingGeo, this.materials.bridge);
    landingMesh.position.set(0, topElevation / 2, landingZ);
    landingMesh.castShadow = true;
    landingMesh.receiveShadow = true;
    this.group.add(landingMesh);

    this.colliders.push({
      type: 'landing',
      minX: -stepWidth / 2,
      maxX: stepWidth / 2,
      minZ: startZ + stepCount * treadDepth,
      maxZ: startZ + stepCount * treadDepth + landingDepth,
      elevation: topElevation,
    });
  }

  /**
   * 2. Ramp & Bridge: Ascending 18° slope, elevated beam, descending slope
   */
  buildRampBridge() {
    const rampLength = 0.220;  // 220mm
    const rampHeight = 0.040;  // 40mm
    const rampWidth = 0.120;   // 120mm (narrower than stairs - tests balance)
    const bridgeLength = 0.200;// 200mm
    const startZ = 0.140;

    // Up-ramp wedge
    const upRampAngle = Math.atan2(rampHeight, rampLength);
    const upRampMeshLength = Math.hypot(rampLength, rampHeight);
    const upRampGeo = new THREE.BoxGeometry(rampWidth, 0.008, upRampMeshLength);
    const upRampMesh = new THREE.Mesh(upRampGeo, this.materials.ramp);
    upRampMesh.position.set(
      0,
      rampHeight / 2,
      startZ + rampLength / 2
    );
    upRampMesh.rotation.x = -upRampAngle;
    upRampMesh.castShadow = true;
    upRampMesh.receiveShadow = true;
    this.group.add(upRampMesh);

    // Elevated bridge plank
    const bridgeGeo = new THREE.BoxGeometry(rampWidth, rampHeight, bridgeLength);
    const bridgeMesh = new THREE.Mesh(bridgeGeo, this.materials.bridge);
    const bridgeCenterZ = startZ + rampLength + bridgeLength / 2;
    bridgeMesh.position.set(0, rampHeight / 2, bridgeCenterZ);
    bridgeMesh.castShadow = true;
    bridgeMesh.receiveShadow = true;
    this.group.add(bridgeMesh);

    // Bridge glowing edges
    const railGeo = new THREE.BoxGeometry(0.006, 0.008, bridgeLength);
    const leftRail = new THREE.Mesh(railGeo, this.materials.stripe);
    leftRail.position.set(-rampWidth / 2, rampHeight + 0.004, bridgeCenterZ);
    const rightRail = new THREE.Mesh(railGeo, this.materials.stripe);
    rightRail.position.set(rampWidth / 2, rampHeight + 0.004, bridgeCenterZ);
    this.group.add(leftRail);
    this.group.add(rightRail);

    // Down-ramp wedge
    const downRampMesh = new THREE.Mesh(upRampGeo, this.materials.ramp);
    const downRampStartZ = startZ + rampLength + bridgeLength;
    downRampMesh.position.set(
      0,
      rampHeight / 2,
      downRampStartZ + rampLength / 2
    );
    downRampMesh.rotation.x = upRampAngle;
    downRampMesh.castShadow = true;
    downRampMesh.receiveShadow = true;
    this.group.add(downRampMesh);

    // Colliders
    this.colliders.push({
      type: 'up_ramp',
      minX: -rampWidth / 2,
      maxX: rampWidth / 2,
      minZ: startZ,
      maxZ: startZ + rampLength,
      startElevation: 0,
      endElevation: rampHeight,
      angleDeg: THREE.MathUtils.radToDeg(upRampAngle)
    });
    this.colliders.push({
      type: 'bridge',
      minX: -rampWidth / 2,
      maxX: rampWidth / 2,
      minZ: startZ + rampLength,
      maxZ: downRampStartZ,
      elevation: rampHeight
    });
    this.colliders.push({
      type: 'down_ramp',
      minX: -rampWidth / 2,
      maxX: rampWidth / 2,
      minZ: downRampStartZ,
      maxZ: downRampStartZ + rampLength,
      startElevation: rampHeight,
      endElevation: 0,
      angleDeg: -THREE.MathUtils.radToDeg(upRampAngle)
    });
  }

  /**
   * 3. Crawl Tunnel: 65mm ceiling (Bittle stands at 88mm, must belly-crawl at ~45mm)
   */
  buildCrawlTunnel() {
    const tunnelLength = 0.280; // 280mm
    const tunnelWidth = 0.160;  // 160mm
    const ceilingHeight = 0.065;// 65mm ceiling!
    const wallThickness = 0.012;
    const startZ = 0.120;
    const centerZ = startZ + tunnelLength / 2;

    // Left wall
    const wallGeo = new THREE.BoxGeometry(wallThickness, ceilingHeight, tunnelLength);
    const leftWall = new THREE.Mesh(wallGeo, this.materials.tunnelArch);
    leftWall.position.set(-tunnelWidth / 2 - wallThickness / 2, ceilingHeight / 2, centerZ);
    leftWall.castShadow = true;
    this.group.add(leftWall);

    // Right wall
    const rightWall = new THREE.Mesh(wallGeo, this.materials.tunnelArch);
    rightWall.position.set(tunnelWidth / 2 + wallThickness / 2, ceilingHeight / 2, centerZ);
    rightWall.castShadow = true;
    this.group.add(rightWall);

    // Roof (semi-transparent so operator can see robot crawl inside)
    const roofGeo = new THREE.BoxGeometry(tunnelWidth + wallThickness * 2, 0.008, tunnelLength);
    const roofMesh = new THREE.Mesh(roofGeo, this.materials.tunnelCeiling);
    roofMesh.position.set(0, ceilingHeight + 0.004, centerZ);
    roofMesh.castShadow = true;
    this.group.add(roofMesh);

    // Warning clearance sign at entry
    const archBorderGeo = new THREE.BoxGeometry(tunnelWidth + 0.02, 0.010, 0.008);
    const archBorder = new THREE.Mesh(archBorderGeo, this.materials.stripe);
    archBorder.position.set(0, ceilingHeight, startZ);
    this.group.add(archBorder);

    this.colliders.push({
      type: 'tunnel',
      minX: -tunnelWidth / 2,
      maxX: tunnelWidth / 2,
      minZ: startZ,
      maxZ: startZ + tunnelLength,
      ceilingHeight: ceilingHeight,
      requiredPosture: 'crouch_or_crawl'
    });
  }

  /**
   * 4. Agility Slalom: Zigzag precision steering cones
   */
  buildAgilitySlalom() {
    const conePositions = [
      { x: 0.045, z: 0.140 },
      { x: -0.045, z: 0.240 },
      { x: 0.045, z: 0.340 },
      { x: -0.045, z: 0.440 }
    ];

    conePositions.forEach((pos, idx) => {
      const coneGroup = new THREE.Group();
      coneGroup.position.set(pos.x, 0, pos.z);

      // Base ring
      const baseGeo = new THREE.CylinderGeometry(0.022, 0.025, 0.005, 16);
      const baseMesh = new THREE.Mesh(baseGeo, this.materials.coneOrange);
      baseMesh.position.y = 0.0025;
      coneGroup.add(baseMesh);

      // Body cone
      const coneGeo = new THREE.ConeGeometry(0.018, 0.055, 16);
      const coneMesh = new THREE.Mesh(coneGeo, this.materials.coneOrange);
      coneMesh.position.y = 0.0275;
      coneMesh.castShadow = true;
      coneGroup.add(coneMesh);

      // White stripe ring
      const ringGeo = new THREE.CylinderGeometry(0.011, 0.013, 0.012, 16);
      const ringMesh = new THREE.Mesh(ringGeo, this.materials.coneWhite);
      ringMesh.position.y = 0.026;
      coneGroup.add(ringMesh);

      this.group.add(coneGroup);

      this.colliders.push({
        type: 'cone',
        x: pos.x,
        z: pos.z,
        radius: 0.022,
        height: 0.055,
        index: idx + 1
      });
    });
  }

  /**
   * Query ground elevation at (x, z) coordinates.
   * Returns elevation in meters (0 if flat ground).
   */
  getElevationAt(x, z) {
    if (this.currentPreset === COURSE_PRESETS.NONE) return 0;

    for (const c of this.colliders) {
      if (c.type === 'step' || c.type === 'landing') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          return c.elevation;
        }
      } else if (c.type === 'up_ramp') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          const ratio = (z - c.minZ) / (c.maxZ - c.minZ);
          return THREE.MathUtils.lerp(c.startElevation, c.endElevation, ratio);
        }
      } else if (c.type === 'bridge') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          return c.elevation;
        }
      } else if (c.type === 'down_ramp') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          const ratio = (z - c.minZ) / (c.maxZ - c.minZ);
          return THREE.MathUtils.lerp(c.startElevation, c.endElevation, ratio);
        }
      }
    }
    return 0;
  }

  /**
   * Check collision of a robot with known bounds or height at (x, y, z)
   */
  checkCollision(x, y, z, robotHeight = 0.088) {
    const issues = [];
    for (const c of this.colliders) {
      if (c.type === 'tunnel') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          const headTopY = y + robotHeight;
          if (headTopY > c.ceilingHeight) {
            issues.push({
              obstacle: 'tunnel_ceiling',
              message: `Robot height ${(headTopY * 1000).toFixed(0)}mm exceeds tunnel ceiling ${(c.ceilingHeight * 1000).toFixed(0)}mm. Crawl posture required.`
            });
          }
        }
      } else if (c.type === 'cone') {
        const dist = Math.hypot(x - c.x, z - c.z);
        if (dist < c.radius + 0.045) { // 45mm is approximate half-width of Bittle body
          issues.push({
            obstacle: `cone_${c.index}`,
            message: `Body collided with slalom cone ${c.index} (dist ${(dist * 1000).toFixed(0)}mm < radius).`
          });
        }
      }
    }
    return {
      collided: issues.length > 0,
      issues: issues
    };
  }

  getLayout() {
    return {
      preset: this.currentPreset,
      obstacles: this.colliders,
    };
  }
}
