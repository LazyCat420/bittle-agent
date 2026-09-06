/**
 * Bittle 3D Digital Twin Viewer
 * High-fidelity Three.js WebGL rendering for Petoi Bittle quadruped.
 */

import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import { OBJLoader } from '/vendor/OBJLoader.js';
import { BUILTIN_MOVESETS } from './builtin_movesets.js';

export { BUILTIN_MOVESETS };

// OpenCat standard poses
export const STAND_POSE = {
  0: 0,    // Head pan
  1: 0,    // Tail
  8: -45,  // Shoulder FL
  9: -45,  // Shoulder FR
  10: -45, // Shoulder BR
  11: -45, // Shoulder BL
  12: 80,  // Knee FL
  13: 80,  // Knee FR
  14: 80,  // Knee BR
  15: 80   // Knee BL
};

export const REST_POSE = {
  0: 0,    // Head pan
  1: 0,    // Tail
  8: -55,  // Shoulder FL
  9: -55,  // Shoulder FR
  10: 55,  // Shoulder BR
  11: 55,  // Shoulder BL
  12: 60,  // Knee FL
  13: 60,  // Knee FR
  14: 60,  // Knee BR
  15: 60   // Knee BL
};

// Slew speed: 320 degrees per second matching SimBackend
const SLEW_DEG_PER_SEC = 320.0;

// Authentic Bittle PBR Materials
const MATERIALS = {
  yellow: new THREE.MeshStandardMaterial({
    color: 0xF5B700,
    roughness: 0.35,
    metalness: 0.05,
    name: "bittle_yellow"
  }),
  blue: new THREE.MeshStandardMaterial({
    color: 0x1E46C7,
    roughness: 0.35,
    metalness: 0.1,
    name: "bittle_blue"
  }),
  red: new THREE.MeshStandardMaterial({
    color: 0xD62828,
    roughness: 0.3,
    metalness: 0.15,
    name: "bittle_red"
  }),
  dark: new THREE.MeshStandardMaterial({
    color: 0x1A1A1A,
    roughness: 0.6,
    metalness: 0.2,
    name: "bittle_dark"
  }),
  highlight: new THREE.MeshStandardMaterial({
    color: 0xFF9900,
    roughness: 0.2,
    metalness: 0.3,
    emissive: 0xFF6600,
    emissiveIntensity: 0.4
  }),
  estop: new THREE.MeshStandardMaterial({
    color: 0xFF2222,
    roughness: 0.4,
    emissive: 0x990000,
    emissiveIntensity: 0.6
  })
};

export class BittleViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) throw new Error(`Container #${containerId} not found`);

    this.currentAngles = { ...STAND_POSE };
    this.targetAngles = { ...STAND_POSE };
    this.jointNodes = {};
    this.interactiveMeshes = [];
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.onSelectCallback = null;
    this.estopEngaged = false;
    this.isLoaded = false;

    // Moveset Sequence Player State
    this.activeSequence = null;
    this.registeredMovesets = {};
    this.currentFrameIdx = 0;
    this.frameTimer = 0;
    this.isPlaying = false;
    this.isLooping = false;
    this.playbackSpeed = 1.0;
    this.onSequenceProgress = null;
    this.onSequenceDone = null;

    this.initScene();
    this.initLighting();
    this.initGround();
    this.initControls();
    this.setupInteraction();

    this.clock = new THREE.Clock();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);

    window.addEventListener('resize', () => this.onWindowResize());
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e1116);
    this.scene.fog = new THREE.FogExp2(0x0e1116, 1.2);

    this.camera = new THREE.PerspectiveCamera(
      45,
      this.container.clientWidth / this.container.clientHeight,
      0.01,
      20
    );
    this.camera.position.set(0.35, 0.28, 0.38);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    this.container.appendChild(this.renderer.domElement);

    // Root robot group - placed so feet touch ground at Y=0 in standing pose (Z = -53.2mm)
    this.robotGroup = new THREE.Group();
    this.robotGroup.position.set(0, 0.0532, 0);
    this.scene.add(this.robotGroup);
  }

  initLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfff5e6, 2.2);
    keyLight.position.set(1.5, 2.5, 1.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 6;
    keyLight.shadow.camera.left = -0.6;
    keyLight.shadow.camera.right = 0.6;
    keyLight.shadow.camera.top = 0.6;
    keyLight.shadow.camera.bottom = -0.6;
    keyLight.shadow.bias = -0.0005;
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x8cb5ff, 1.0);
    fillLight.position.set(-1.5, 1.5, -1.5);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffe0a0, 0.7);
    rimLight.position.set(0, -1, 1);
    this.scene.add(rimLight);
  }

  initGround() {
    const grid = new THREE.GridHelper(2, 20, 0x30363d, 0x1f242c);
    grid.position.y = -0.001;
    this.scene.add(grid);

    const planeGeo = new THREE.PlaneGeometry(10, 10);
    const planeMat = new THREE.MeshStandardMaterial({
      color: 0x090c10,
      roughness: 0.9,
      metalness: 0.1
    });
    const ground = new THREE.Mesh(planeGeo, planeMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.002;
    ground.receiveShadow = true;
    this.scene.add(ground);
  }

  initControls() {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 0.06, 0);
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // prevent going beneath floor
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 1.5;
  }

  setupInteraction() {
    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects(this.interactiveMeshes, true);

      if (intersects.length > 0) {
        let node = intersects[0].object;
        while (node && node !== this.robotGroup) {
          if (node.userData && node.userData.jointId !== undefined) {
            if (this.onSelectCallback) {
              this.onSelectCallback(node.userData.jointId, node.userData.name);
            }
            break;
          }
          node = node.parent;
        }
      }
    });
  }

  async loadModel(assetsPath = '/assets/meshes/') {
    const loader = new OBJLoader();
    const meshNames = [
      'base_link.obj', 'battery_1.obj', 'cover_1.obj', 'front__1.obj', 'rear__1.obj',
      'c_neck__1.obj', 'servo_neck__1.obj', 'head__1.obj', 'jaw_1.obj',
      'servo_rfs_1.obj', 'c_thrf__1.obj', 'th_rf_1.obj', 'tho_rf__1.obj', 'c_thorf_1.obj', 'tube__1.obj', 'servos_rf_1.obj', 'shank_rf_1.obj',
      'servo_lfs_1.obj', 'c_thlf_1.obj', 'th_lf_1.obj', 'tho_lf_1.obj', 'c_tholf_1.obj', 'tube_lf_1.obj', 'servos_lf_1.obj', 'shank_lf_1.obj',
      'servo_rrs__1.obj', 'c_thrr_1.obj', 'th_rr_1.obj', 'tho_rr_1.obj', 'c_thorr_1.obj', 'tube_rr_1.obj', 'servos_rr_1.obj', 'shank_rr_1.obj',
      'servo_lrs__1.obj', 'c_thlr_1.obj', 'th_lr__1.obj', 'tho_lr_1.obj', 'c_tholr__1.obj', 'tube_lr_1.obj', 'servos_lr_1.obj', 'shank_lr_1.obj'
    ];

    const geometries = {};
    const loadPromises = meshNames.map(name => {
      return new Promise((resolve) => {
        loader.load(
          `${assetsPath}${name}`,
          (obj) => {
            obj.traverse((child) => {
              if (child.isMesh) {
                geometries[name] = child.geometry;
                child.geometry.computeVertexNormals();
              }
            });
            resolve();
          },
          undefined,
          () => {
            console.warn(`Could not load ${name}, falling back to placeholder`);
            resolve();
          }
        );
      });
    });

    await Promise.all(loadPromises);
    this.buildHierarchy(geometries);
    this.isLoaded = true;
  }

  createPartMesh(geometry, material, userData = {}, pivot = null) {
    if (!geometry) {
      // Fallback box
      geometry = new THREE.BoxGeometry(0.02, 0.02, 0.02);
    }
    // Clone so each mesh has independent buffer transforms
    const geo = geometry.clone();
    // Convert CAD millimeter coordinates to meters
    geo.scale(0.001, 0.001, 0.001);

    // If a pivot is provided, translate the vertex buffer by -pivot
    // so the mesh is centered at its joint rotation axis
    if (pivot) {
      geo.translate(-pivot.x, -pivot.y, -pivot.z);
    }

    const mesh = new THREE.Mesh(geo, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData = userData;
    this.interactiveMeshes.push(mesh);
    return mesh;
  }

  buildHierarchy(geos) {
    // ─── 1. Torso Chassis ──────────────────────────────────────
    const torso = new THREE.Group();
    torso.userData = { name: "Torso" };

    // Rigid chassis parts in torso CAD coordinates (pivot = null)
    if (geos['base_link.obj']) torso.add(this.createPartMesh(geos['base_link.obj'], MATERIALS.yellow));
    if (geos['front__1.obj']) torso.add(this.createPartMesh(geos['front__1.obj'], MATERIALS.blue));
    if (geos['rear__1.obj']) torso.add(this.createPartMesh(geos['rear__1.obj'], MATERIALS.blue));
    if (geos['cover_1.obj']) torso.add(this.createPartMesh(geos['cover_1.obj'], MATERIALS.dark));
    if (geos['battery_1.obj']) torso.add(this.createPartMesh(geos['battery_1.obj'], MATERIALS.dark));
    if (geos['c_neck__1.obj']) torso.add(this.createPartMesh(geos['c_neck__1.obj'], MATERIALS.blue));

    // Fixed shoulder servo housings mounted directly to torso chassis
    if (geos['servo_rfs_1.obj']) torso.add(this.createPartMesh(geos['servo_rfs_1.obj'], MATERIALS.dark));
    if (geos['servo_lfs_1.obj']) torso.add(this.createPartMesh(geos['servo_lfs_1.obj'], MATERIALS.dark));
    if (geos['servo_rrs__1.obj']) torso.add(this.createPartMesh(geos['servo_rrs__1.obj'], MATERIALS.dark));
    if (geos['servo_lrs__1.obj']) torso.add(this.createPartMesh(geos['servo_lrs__1.obj'], MATERIALS.dark));

    // ─── 2. Head Assembly (Joint 0) ────────────────────────────
    const neckPivotPos = new THREE.Vector3(0.047554, 0.0, 0.035941);
    const headPivot = new THREE.Group();
    headPivot.position.copy(neckPivotPos);
    headPivot.userData = { jointId: 0, name: "Head Pan" };

    if (geos['servo_neck__1.obj']) {
      headPivot.add(this.createPartMesh(geos['servo_neck__1.obj'], MATERIALS.dark, { jointId: 0 }, neckPivotPos));
    }
    if (geos['jaw_1.obj']) {
      headPivot.add(this.createPartMesh(geos['jaw_1.obj'], MATERIALS.blue, { jointId: 0 }, neckPivotPos));
    }
    if (geos['head__1.obj']) {
      headPivot.add(this.createPartMesh(geos['head__1.obj'], MATERIALS.yellow, { jointId: 0 }, neckPivotPos));
    }

    torso.add(headPivot);
    // Head pan rotates horizontally around torso local Z axis (+Z is upward)
    this.jointNodes[0] = { node: headPivot, axis: new THREE.Vector3(0, 0, 1), dir: 1 };

    // ─── 3. Tail Assembly (Joint 1) ────────────────────────────
    const tailPivot = new THREE.Group();
    tailPivot.position.set(-0.065, 0, 0.025);
    tailPivot.userData = { jointId: 1, name: "Tail Wag" };
    torso.add(tailPivot);
    this.jointNodes[1] = { node: tailPivot, axis: new THREE.Vector3(0, 0, 1), dir: 1 };

    // ─── 4. Articulated Leg Builder Helper ──────────────────────
    const buildLeg = (cfg) => {
      const {
        shoulderJointId, kneeJointId, name,
        shoulderPos, kneeWorldPos,
        thighGeos, kneeServoGeo, shankGeo,
        dir
      } = cfg;

      // Shoulder Pivot
      const shoulderPivot = new THREE.Group();
      shoulderPivot.position.copy(shoulderPos);
      shoulderPivot.userData = { jointId: shoulderJointId, name: `${name} Shoulder` };

      // Thigh Assembly (centered at shoulderPos)
      thighGeos.forEach(g => {
        if (g) {
          shoulderPivot.add(this.createPartMesh(g, MATERIALS.red, { jointId: shoulderJointId }, shoulderPos));
        }
      });

      // Knee Pivot (relative to shoulderPos)
      const kneePosRel = new THREE.Vector3().subVectors(kneeWorldPos, shoulderPos);
      const kneePivot = new THREE.Group();
      kneePivot.position.copy(kneePosRel);
      kneePivot.userData = { jointId: kneeJointId, name: `${name} Knee` };

      // Knee Servo (centered at kneeWorldPos)
      if (kneeServoGeo) {
        kneePivot.add(this.createPartMesh(kneeServoGeo, MATERIALS.dark, { jointId: kneeJointId }, kneeWorldPos));
      }

      // Shank / Lower Leg Bracket (centered at kneeWorldPos)
      if (shankGeo) {
        kneePivot.add(this.createPartMesh(shankGeo, MATERIALS.blue, { jointId: kneeJointId }, kneeWorldPos));
      }

      shoulderPivot.add(kneePivot);
      torso.add(shoulderPivot);

      // Pitch rotation around torso Y axis (left-right axis)
      this.jointNodes[shoulderJointId] = { node: shoulderPivot, axis: new THREE.Vector3(0, 1, 0), dir: dir };
      this.jointNodes[kneeJointId] = { node: kneePivot, axis: new THREE.Vector3(0, 1, 0), dir: dir };
    };

    // ─── Front-Right Leg (Joints 9 & 13) ────────────────────────
    buildLeg({
      shoulderJointId: 9, kneeJointId: 13, name: "Right Front",
      shoulderPos: new THREE.Vector3(0.0525, -0.0485, 0.022),
      kneeWorldPos: new THREE.Vector3(0.008067, -0.04708, 0.010094),
      thighGeos: [geos['c_thrf__1.obj'], geos['th_rf_1.obj'], geos['tho_rf__1.obj'], geos['c_thorf_1.obj'], geos['tube__1.obj']],
      kneeServoGeo: geos['servos_rf_1.obj'],
      shankGeo: geos['shank_rf_1.obj'],
      dir: 1
    });

    // ─── Front-Left Leg (Joints 8 & 12) ─────────────────────────
    buildLeg({
      shoulderJointId: 8, kneeJointId: 12, name: "Left Front",
      shoulderPos: new THREE.Vector3(0.0525, 0.0485, 0.022),
      kneeWorldPos: new THREE.Vector3(0.008067, 0.04708, 0.010094),
      thighGeos: [geos['c_thlf_1.obj'], geos['th_lf_1.obj'], geos['tho_lf_1.obj'], geos['c_tholf_1.obj'], geos['tube_lf_1.obj']],
      kneeServoGeo: geos['servos_lf_1.obj'],
      shankGeo: geos['shank_lf_1.obj'],
      dir: 1
    });

    // ─── Rear-Right Leg (Joints 10 & 14) ────────────────────────
    buildLeg({
      shoulderJointId: 10, kneeJointId: 14, name: "Right Rear",
      shoulderPos: new THREE.Vector3(-0.0525, -0.0485, 0.022),
      kneeWorldPos: new THREE.Vector3(-0.096933, -0.04708, 0.010094),
      thighGeos: [geos['c_thrr_1.obj'], geos['th_rr_1.obj'], geos['tho_rr_1.obj'], geos['c_thorr_1.obj'], geos['tube_rr_1.obj']],
      kneeServoGeo: geos['servos_rr_1.obj'],
      shankGeo: geos['shank_rr_1.obj'],
      dir: 1
    });

    // ─── Rear-Left Leg (Joints 11 & 15) ─────────────────────────
    buildLeg({
      shoulderJointId: 11, kneeJointId: 15, name: "Left Rear",
      shoulderPos: new THREE.Vector3(-0.0525, 0.0485, 0.022),
      kneeWorldPos: new THREE.Vector3(-0.096933, 0.047082, 0.010094),
      thighGeos: [geos['c_thlr_1.obj'], geos['th_lr__1.obj'], geos['tho_lr_1.obj'], geos['c_tholr__1.obj'], geos['tube_lr_1.obj']],
      kneeServoGeo: geos['servos_lr_1.obj'],
      shankGeo: geos['shank_lr_1.obj'],
      dir: 1
    });

    // Rotate robot so +Z points up and +X forward in standard view
    torso.rotation.x = -Math.PI / 2;
    torso.rotation.z = Math.PI / 2;
    this.robotGroup.add(torso);

    this.applyAngles(STAND_POSE);
  }

  setJointTarget(jointId, angleDeg) {
    jointId = Number(jointId);
    this.targetAngles[jointId] = angleDeg;
  }

  setPose(angles, force = false) {
    if (this.isPlaying && this.activeSequence && !force) return;
    Object.entries(angles).forEach(([k, v]) => {
      this.setJointTarget(Number(k), Number(v));
    });
  }

  resetPose(preset = 'stand') {
    this.isPlaying = false;
    this.setPose(preset === 'rest' ? REST_POSE : STAND_POSE, true);
  }

  setEstop(engaged) {
    if (this.estopEngaged === engaged) return;
    this.estopEngaged = engaged;
    if (engaged) {
      // Robot drops limp
      this.isPlaying = false;
      this.setPose(REST_POSE, true);
      this.robotGroup.position.y = 0.02; // sink to floor
    } else {
      this.robotGroup.position.y = 0.0532;
      this.setPose(STAND_POSE, true);
    }
  }

  registerMovesets(movesets) {
    if (!movesets) return;
    const list = Array.isArray(movesets) ? movesets : Object.values(movesets);
    list.forEach(m => {
      if (m && m.name) {
        this.registeredMovesets[m.name] = m;
      }
    });
  }

  setCameraPreset(name) {
    switch (name) {
      case 'iso':
        this.camera.position.set(0.35, 0.28, 0.38);
        this.controls.target.set(0, 0.06, 0);
        break;
      case 'front':
        this.camera.position.set(0, 0.12, 0.45);
        this.controls.target.set(0, 0.06, 0);
        break;
      case 'side':
        this.camera.position.set(0.45, 0.12, 0);
        this.controls.target.set(0, 0.06, 0);
        break;
      case 'top':
        this.camera.position.set(0, 0.55, 0.01);
        this.controls.target.set(0, 0.06, 0);
        break;
    }
    this.controls.update();
  }

  setWireframe(enabled) {
    Object.values(MATERIALS).forEach(m => {
      m.wireframe = enabled;
    });
  }

  applyAngles(angles) {
    Object.entries(angles).forEach(([jointIdStr, deg]) => {
      const jointId = Number(jointIdStr);
      const joint = this.jointNodes[jointId];
      if (joint && joint.node) {
        const rad = THREE.MathUtils.degToRad(deg) * joint.dir;
        joint.node.setRotationFromAxisAngle(joint.axis, rad);
      }
    });
  }

  playSequence(sequenceOrName, options = {}) {
    let frames = null;
    let name = options.name || 'Sequence';
    let description = options.description || '';

    if (typeof sequenceOrName === 'string') {
      const b = this.registeredMovesets[sequenceOrName] || BUILTIN_MOVESETS[sequenceOrName];
      if (b) {
        frames = b.frames;
        name = b.label || b.name;
        description = b.description || '';
      }
    } else if (Array.isArray(sequenceOrName)) {
      frames = sequenceOrName;
    } else if (sequenceOrName && Array.isArray(sequenceOrName.frames)) {
      frames = sequenceOrName.frames;
      name = sequenceOrName.name || name;
      description = sequenceOrName.description || description;
    }

    if (!frames || frames.length === 0) {
      console.warn('playSequence: No valid frames found', sequenceOrName);
      return false;
    }

    this.activeSequence = {
      name,
      description,
      frames: frames.map(f => ({
        angles: { ...(f.angles || {}) },
        speed_deg_per_step: f.speed_deg_per_step || 8,
        delay_ms: f.delay_ms || 200,
      }))
    };

    if (options.loop !== undefined) {
      this.isLooping = Boolean(options.loop);
    }
    this.currentFrameIdx = 0;
    this.frameTimer = 0;
    this.isPlaying = true;
    this.applyFrame(0, false);
    this.notifyProgress();
    return true;
  }

  applyFrame(idx, snap = false) {
    if (!this.activeSequence || !this.activeSequence.frames[idx]) return;
    const frame = this.activeSequence.frames[idx];
    const angles = frame.angles;
    this.currentFrameIdx = idx;

    Object.entries(angles).forEach(([k, v]) => {
      const id = Number(k);
      const val = Number(v);
      this.targetAngles[id] = val;
      if (snap) {
        this.currentAngles[id] = val;
      }
    });

    if (snap) {
      this.applyAngles(this.currentAngles);
    }
  }

  pauseSequence() {
    this.isPlaying = false;
    this.notifyProgress();
  }

  resumeSequence() {
    if (!this.activeSequence) return;
    this.isPlaying = true;
    this.notifyProgress();
  }

  togglePlayPause() {
    if (this.isPlaying) {
      this.pauseSequence();
    } else {
      this.resumeSequence();
    }
  }

  stopSequence() {
    this.isPlaying = false;
    this.currentFrameIdx = 0;
    this.frameTimer = 0;
    this.applyFrame(0, false);
    this.notifyProgress();
  }

  seekFrame(idx) {
    if (!this.activeSequence) return;
    const clamped = Math.max(0, Math.min(idx, this.activeSequence.frames.length - 1));
    this.applyFrame(clamped, true);
    this.notifyProgress();
  }

  seekFraction(frac) {
    if (!this.activeSequence) return;
    const totalFrames = this.activeSequence.frames.length;
    const idx = Math.min(Math.floor(frac * totalFrames), totalFrames - 1);
    this.seekFrame(idx);
  }

  stepForward() {
    if (!this.activeSequence) return;
    this.pauseSequence();
    const nextIdx = (this.currentFrameIdx + 1) % this.activeSequence.frames.length;
    this.seekFrame(nextIdx);
  }

  stepBackward() {
    if (!this.activeSequence) return;
    this.pauseSequence();
    const prevIdx = (this.currentFrameIdx - 1 + this.activeSequence.frames.length) % this.activeSequence.frames.length;
    this.seekFrame(prevIdx);
  }

  setPlaybackSpeed(speed) {
    this.playbackSpeed = Number(speed) || 1.0;
    this.notifyProgress();
  }

  setLoop(loop) {
    this.isLooping = Boolean(loop);
    this.notifyProgress();
  }

  notifyProgress() {
    if (!this.onSequenceProgress) return;
    const totalFrames = this.activeSequence ? this.activeSequence.frames.length : 0;
    const currentFrame = this.currentFrameIdx;
    const currentFrameData = this.activeSequence ? this.activeSequence.frames[currentFrame] : null;

    let totalMs = 0;
    let currentMs = 0;
    if (this.activeSequence) {
      this.activeSequence.frames.forEach((f, i) => {
        const dur = f.delay_ms || 200;
        if (i < currentFrame) currentMs += dur;
        else if (i === currentFrame) currentMs += this.frameTimer * 1000;
        totalMs += dur;
      });
    }

    this.onSequenceProgress({
      sequenceName: this.activeSequence ? this.activeSequence.name : 'No active sequence',
      description: this.activeSequence ? this.activeSequence.description : '',
      currentFrameIndex: currentFrame,
      totalFrames: totalFrames,
      currentTimeSec: currentMs / 1000,
      totalTimeSec: totalMs / 1000,
      isPlaying: this.isPlaying,
      isLooping: this.isLooping,
      speed: this.playbackSpeed,
      angles: currentFrameData ? currentFrameData.angles : this.currentAngles,
    });
  }

  animate() {
    requestAnimationFrame(this.animate);
    const dt = Math.min(this.clock.getDelta(), 0.1);

    // Sequence advance logic
    if (this.isLoaded && this.isPlaying && this.activeSequence) {
      const currentFrame = this.activeSequence.frames[this.currentFrameIdx];
      const targetDelaySec = ((currentFrame.delay_ms || 200) / 1000) / this.playbackSpeed;

      let jointsSettled = true;
      Object.keys(currentFrame.angles).forEach(k => {
        const id = Number(k);
        if (Math.abs(this.targetAngles[id] - this.currentAngles[id]) > 3.0) {
          jointsSettled = false;
        }
      });

      this.frameTimer += dt;

      if (this.frameTimer >= targetDelaySec && jointsSettled) {
        this.frameTimer = 0;
        if (this.currentFrameIdx < this.activeSequence.frames.length - 1) {
          this.applyFrame(this.currentFrameIdx + 1, false);
          this.notifyProgress();
        } else {
          if (this.isLooping) {
            this.applyFrame(0, false);
            this.notifyProgress();
          } else {
            this.isPlaying = false;
            this.notifyProgress();
            if (this.onSequenceDone) this.onSequenceDone();
          }
        }
      }
    }

    // Servo slew interpolation
    if (this.isLoaded) {
      let changed = false;
      const maxDelta = SLEW_DEG_PER_SEC * dt;

      Object.keys(this.targetAngles).forEach(k => {
        const id = Number(k);
        const target = this.targetAngles[id];
        const cur = this.currentAngles[id];
        if (Math.abs(target - cur) > 0.1) {
          const step = Math.sign(target - cur) * Math.min(Math.abs(target - cur), maxDelta);
          this.currentAngles[id] = cur + step;
          changed = true;
        } else {
          this.currentAngles[id] = target;
        }
      });

      if (changed) {
        this.applyAngles(this.currentAngles);
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  onWindowResize() {
    if (!this.container) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
