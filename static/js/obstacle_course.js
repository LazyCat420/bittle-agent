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
        color: 0x2a3240, // Industrial slate titanium
        roughness: 0.55,
        metalness: 0.3,
      }),
      riser: new THREE.MeshStandardMaterial({
        color: 0x1a212d, // Dark charcoal riser
        roughness: 0.65,
        metalness: 0.25,
      }),
      stripe: new THREE.MeshStandardMaterial({
        color: 0xffaa00, // High-visibility hazard amber
        emissive: 0xff7700,
        emissiveIntensity: 0.6,
        roughness: 0.25,
        metalness: 0.1,
      }),
      ramp: new THREE.MeshStandardMaterial({
        color: 0x334155, // Slate blue grip deck
        roughness: 0.5,
        metalness: 0.3,
      }),
      bridge: new THREE.MeshStandardMaterial({
        color: 0x1e293b, // Deep aerospace slate
        roughness: 0.4,
        metalness: 0.5,
      }),
      bridgeRail: new THREE.MeshStandardMaterial({
        color: 0x38bdf8, // Radiant cyan guidance rail
        emissive: 0x0284c7,
        emissiveIntensity: 0.7,
        roughness: 0.2,
        metalness: 0.1,
      }),
      tunnelArch: new THREE.MeshStandardMaterial({
        color: 0x3b4454,
        roughness: 0.45,
        metalness: 0.4,
        side: THREE.DoubleSide,
      }),
      tunnelCeiling: new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        emissive: 0x0284c7,
        emissiveIntensity: 0.4,
        roughness: 0.3,
        metalness: 0.2,
        transparent: true,
        opacity: 0.7,
      }),
      coneOrange: new THREE.MeshStandardMaterial({
        color: 0xff5500,
        roughness: 0.35,
        metalness: 0.1,
        emissive: 0xdd3300,
        emissiveIntensity: 0.35,
      }),
      coneWhite: new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.2,
        metalness: 0.1,
        emissive: 0xaaaaaa,
        emissiveIntensity: 0.2,
      }),
      waypoint: new THREE.MeshStandardMaterial({
        color: 0x22c55e,
        emissive: 0x16a34a,
        emissiveIntensity: 0.7,
        transparent: true,
        opacity: 0.6,
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

  /**
   * A benchmark rollout's terrain (bittle.rollout.v1 source.field): the boxes the physics engine
   * placed, in MuJoCo world metres (x forward, y left, z up; floor at plane_z). Viewer axes:
   * x -> x, z -> y (up), y -> -z. The box tops sit at pos.z + half.z above plane_z.
   */
  loadField(field) {
    this.clear();
    if (!field || !Array.isArray(field.boxes) || field.boxes.length === 0) return;
    this.currentPreset = 'replay_field';
    const planeZ = Number.isFinite(field.plane_z) ? field.plane_z : -0.01;
    const rock = this.materials.rock || (this.materials.rock = new THREE.MeshStandardMaterial({ color: 0x8b7355, roughness: 0.85, metalness: 0.05 }));
    const top = this.materials.rockTop || (this.materials.rockTop = new THREE.MeshStandardMaterial({ color: 0xb59a6b, roughness: 0.8, metalness: 0.05 }));
    for (const b of field.boxes) {
      const [mx, my, mz] = b.pos;
      const [hx, hy, hz] = b.half;
      const topZ = mz + hz - planeZ;           // height of the box top above the floor (m)
      if (topZ <= 0) continue;
      const yawRad = -THREE.MathUtils.degToRad(b.yaw_deg || 0);  // y flips, so the yaw flips
      // draw only the part above the floor: a slab of height topZ whose centre is topZ/2 above the floor
      const geo = new THREE.BoxGeometry(2 * hx, topZ, 2 * hy);
      const mesh = new THREE.Mesh(geo, [rock, rock, top, rock, rock, rock]);
      mesh.position.set(mx, topZ / 2 + 0.0005, -my);
      mesh.rotation.y = yawRad;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.group.add(mesh);
      const r = Math.hypot(hx, hy);
      this.colliders.push({ type: 'rock', minX: mx - r, maxX: mx + r, minZ: -my - r, maxZ: -my + r, elevation: topZ, yaw: yawRad, hx, hy, x: mx, z: -my });
    }
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
   * Generated along +X forward axis (in front of Bittle's nose).
   * Each step: rise = 18mm (0.018m), tread = 80mm (0.08m), width = 240mm (0.24m) along Z.
   */
  buildMiniStairs() {
    const stepCount = 3;
    const riserHeight = 0.018; // 18mm
    const treadDepth = 0.080;  // 80mm along X
    const stepWidth = 0.240;   // 240mm along Z
    const startX = 0.140;      // 14cm in front of initial spawn

    for (let i = 0; i < stepCount; i++) {
      const currentElevation = (i + 1) * riserHeight;
      const centerX = startX + i * treadDepth + treadDepth / 2;
      const centerY = currentElevation / 2;

      // Solid block supporting the step (depth along X, elevation Y, width along Z)
      const stepGeo = new THREE.BoxGeometry(treadDepth, currentElevation, stepWidth);
      const stepMesh = new THREE.Mesh(stepGeo, this.materials.tread);
      stepMesh.position.set(centerX, centerY, 0);
      stepMesh.castShadow = true;
      stepMesh.receiveShadow = true;
      this.group.add(stepMesh);

      // Warning hazard stripe on step nosing (edge facing incoming robot)
      const stripeGeo = new THREE.BoxGeometry(0.006, 0.002, stepWidth);
      const stripeMesh = new THREE.Mesh(stripeGeo, this.materials.stripe);
      stripeMesh.position.set(startX + i * treadDepth + 0.003, currentElevation + 0.001, 0);
      this.group.add(stripeMesh);

      // Record collider info
      this.colliders.push({
        type: 'step',
        minX: startX + i * treadDepth,
        maxX: startX + (i + 1) * treadDepth,
        minZ: -stepWidth / 2,
        maxZ: stepWidth / 2,
        elevation: currentElevation,
        riser: riserHeight,
        index: i + 1,
      });
    }

    // Top landing platform
    const landingDepth = 0.140;
    const topElevation = stepCount * riserHeight;
    const landingX = startX + stepCount * treadDepth + landingDepth / 2;
    const landingGeo = new THREE.BoxGeometry(landingDepth, topElevation, stepWidth);
    const landingMesh = new THREE.Mesh(landingGeo, this.materials.bridge);
    landingMesh.position.set(landingX, topElevation / 2, 0);
    landingMesh.castShadow = true;
    landingMesh.receiveShadow = true;
    this.group.add(landingMesh);

    this.colliders.push({
      type: 'landing',
      minX: startX + stepCount * treadDepth,
      maxX: startX + stepCount * treadDepth + landingDepth,
      minZ: -stepWidth / 2,
      maxZ: stepWidth / 2,
      elevation: topElevation,
    });
  }

  /**
   * 2. Ramp & Bridge: Ascending 10.3° slope, elevated beam, descending slope
   * Generated along +X forward axis in front of Bittle.
   */
  buildRampBridge() {
    const rampLength = 0.220;  // 220mm along X
    const rampHeight = 0.040;  // 40mm along Y
    const rampWidth = 0.140;   // 140mm along Z
    const bridgeLength = 0.200;// 200mm along X
    const startX = 0.130;

    // Up-ramp wedge (slanted along X, tilted around Z axis)
    const upRampAngle = Math.atan2(rampHeight, rampLength);
    const upRampMeshLength = Math.hypot(rampLength, rampHeight);
    const upRampGeo = new THREE.BoxGeometry(upRampMeshLength, 0.008, rampWidth);
    const upRampMesh = new THREE.Mesh(upRampGeo, this.materials.ramp);
    upRampMesh.position.set(
      startX + rampLength / 2,
      rampHeight / 2,
      0
    );
    upRampMesh.rotation.z = upRampAngle;
    upRampMesh.castShadow = true;
    upRampMesh.receiveShadow = true;
    this.group.add(upRampMesh);

    // Elevated bridge plank
    const bridgeGeo = new THREE.BoxGeometry(bridgeLength, rampHeight, rampWidth);
    const bridgeMesh = new THREE.Mesh(bridgeGeo, this.materials.bridge);
    const bridgeCenterX = startX + rampLength + bridgeLength / 2;
    bridgeMesh.position.set(bridgeCenterX, rampHeight / 2, 0);
    bridgeMesh.castShadow = true;
    bridgeMesh.receiveShadow = true;
    this.group.add(bridgeMesh);

    // Bridge glowing edges (along both lateral sides of bridge)
    const railGeo = new THREE.BoxGeometry(bridgeLength, 0.008, 0.006);
    const leftRail = new THREE.Mesh(railGeo, this.materials.bridgeRail);
    leftRail.position.set(bridgeCenterX, rampHeight + 0.004, -rampWidth / 2);
    const rightRail = new THREE.Mesh(railGeo, this.materials.bridgeRail);
    rightRail.position.set(bridgeCenterX, rampHeight + 0.004, rampWidth / 2);
    this.group.add(leftRail);
    this.group.add(rightRail);

    // Down-ramp wedge
    const downRampMesh = new THREE.Mesh(upRampGeo, this.materials.ramp);
    const downRampStartX = startX + rampLength + bridgeLength;
    downRampMesh.position.set(
      downRampStartX + rampLength / 2,
      rampHeight / 2,
      0
    );
    downRampMesh.rotation.z = -upRampAngle;
    downRampMesh.castShadow = true;
    downRampMesh.receiveShadow = true;
    this.group.add(downRampMesh);

    // Colliders
    this.colliders.push({
      type: 'up_ramp',
      minX: startX,
      maxX: startX + rampLength,
      minZ: -rampWidth / 2,
      maxZ: rampWidth / 2,
      startElevation: 0,
      endElevation: rampHeight,
      angleDeg: THREE.MathUtils.radToDeg(upRampAngle)
    });
    this.colliders.push({
      type: 'bridge',
      minX: startX + rampLength,
      maxX: downRampStartX,
      minZ: -rampWidth / 2,
      maxZ: rampWidth / 2,
      elevation: rampHeight
    });
    this.colliders.push({
      type: 'down_ramp',
      minX: downRampStartX,
      maxX: downRampStartX + rampLength,
      minZ: -rampWidth / 2,
      maxZ: rampWidth / 2,
      startElevation: rampHeight,
      endElevation: 0,
      angleDeg: -THREE.MathUtils.radToDeg(upRampAngle)
    });
  }

  /**
   * 3. Crawl Tunnel: 65mm ceiling (Bittle stands at 88mm, must belly-crawl at ~45mm)
   * Generated along +X forward axis.
   */
  buildCrawlTunnel() {
    const tunnelLength = 0.280; // 280mm along X
    const tunnelWidth = 0.160;  // 160mm along Z
    const ceilingHeight = 0.065;// 65mm ceiling
    const wallThickness = 0.012;
    const startX = 0.120;
    const centerX = startX + tunnelLength / 2;

    // Left wall (at -Z)
    const wallGeo = new THREE.BoxGeometry(tunnelLength, ceilingHeight, wallThickness);
    const leftWall = new THREE.Mesh(wallGeo, this.materials.tunnelArch);
    leftWall.position.set(centerX, ceilingHeight / 2, -tunnelWidth / 2 - wallThickness / 2);
    leftWall.castShadow = true;
    this.group.add(leftWall);

    // Right wall (at +Z)
    const rightWall = new THREE.Mesh(wallGeo, this.materials.tunnelArch);
    rightWall.position.set(centerX, ceilingHeight / 2, tunnelWidth / 2 + wallThickness / 2);
    rightWall.castShadow = true;
    this.group.add(rightWall);

    // Roof (semi-transparent so operator can see robot crawl inside)
    const roofGeo = new THREE.BoxGeometry(tunnelLength, 0.008, tunnelWidth + wallThickness * 2);
    const roofMesh = new THREE.Mesh(roofGeo, this.materials.tunnelCeiling);
    roofMesh.position.set(centerX, ceilingHeight + 0.004, 0);
    roofMesh.castShadow = true;
    this.group.add(roofMesh);

    // Warning clearance sign at entry portal
    const archBorderGeo = new THREE.BoxGeometry(0.008, 0.010, tunnelWidth + 0.02);
    const archBorder = new THREE.Mesh(archBorderGeo, this.materials.stripe);
    archBorder.position.set(startX, ceilingHeight, 0);
    this.group.add(archBorder);

    this.colliders.push({
      type: 'tunnel',
      minX: startX,
      maxX: startX + tunnelLength,
      minZ: -tunnelWidth / 2,
      maxZ: tunnelWidth / 2,
      ceilingHeight: ceilingHeight,
      requiredPosture: 'crouch_or_crawl'
    });
  }

  /**
   * 4. Agility Slalom: Zigzag precision steering cones
   * Generated along +X forward axis with alternating lateral offsets.
   */
  buildAgilitySlalom() {
    const conePositions = [
      { x: 0.140, z: 0.045 },
      { x: 0.240, z: -0.045 },
      { x: 0.340, z: 0.045 },
      { x: 0.440, z: -0.045 }
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
      if (c.type === 'rock') {
        // rotate the query into the box frame (yaw about the up axis)
        const dx = x - c.x, dz = z - c.z;
        const cs = Math.cos(-c.yaw), sn = Math.sin(-c.yaw);
        const lx = dx * cs - dz * sn, lz = dx * sn + dz * cs;
        if (Math.abs(lx) <= c.hx && Math.abs(lz) <= c.hy) return c.elevation;
      } else if (c.type === 'step' || c.type === 'landing') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          return c.elevation;
        }
      } else if (c.type === 'up_ramp') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          const ratio = (x - c.minX) / (c.maxX - c.minX);
          return THREE.MathUtils.lerp(c.startElevation, c.endElevation, ratio);
        }
      } else if (c.type === 'bridge') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          return c.elevation;
        }
      } else if (c.type === 'down_ramp') {
        if (x >= c.minX && x <= c.maxX && z >= c.minZ && z <= c.maxZ) {
          const ratio = (x - c.minX) / (c.maxX - c.minX);
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
