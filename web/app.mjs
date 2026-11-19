// app.mjs -- the browser demo: MuJoCo (WASM) + the policy (onnxruntime-web) in sim.mjs, drawn with three.js.
// Tap the floor: the robot turns, walks there and stops. Buttons and arrow keys drive it directly.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.17.0/mujoco.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';
import { createSim, goToAxes } from './sim.mjs';

const $ = id => document.getElementById(id), say = t => { $('progress').textContent = t; };
const KEYS = { up: [1, 0, 0], down: [-1, 0, 0], left: [0, 0, 1], right: [0, 0, -1], sleft: [0, 1, 0], sright: [0, -1, 0], stop: [0, 0, 0] };

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1;                     // GitHub Pages sends no cross-origin isolation headers
const load = async p => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
const mujoco = await loadMujoco();
say('Building Chhaya…');
const sim = await createSim(mujoco, load);
say('Loading how Chhaya walks…');
const session = await ort.InferenceSession.create('assets/policy.onnx', { executionProviders: ['wasm'] });
const policy = async o => (await session.run({ obs: new ort.Tensor('float32', sim.norm(o), [1, o.length]) })).action.data;

// ---- scene: pale floor with a faint grid, soft key light (MuJoCo is z-up, so three.js is too) ----
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#f5f5f7'); scene.fog = new THREE.Fog('#f5f5f7', 4, 12);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
camera.up.set(0, 0, 1); camera.position.set(1.15, -0.95, 0.7);          // front three-quarter (the robot faces +x)
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.25); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.4; controls.maxDistance = 5;
scene.add(new THREE.HemisphereLight('#ffffff', '#d8d8de', 1.6));
const sun = new THREE.DirectionalLight('#ffffff', 2.2);
sun.position.set(1.5, -2, 3); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.1, far: 8 });
scene.add(sun, sun.target);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: '#ededf0', roughness: 0.95 }));
floor.receiveShadow = true; scene.add(floor);
const grid = new THREE.GridHelper(40, 160, '#d4d4da', '#e2e2e7');
grid.rotation.x = Math.PI / 2; grid.position.z = 0.0005; scene.add(grid);

// ---- the robot: one three.js mesh per visible MuJoCo geom, posed from geom_xpos / geom_xmat ----
const { m, d } = sim, geoms = [];
for (let g = 0; g < m.ngeom; g++) {
  const type = m.geom_type[g], s = m.geom_size.subarray(3 * g, 3 * g + 3);
  if (m.geom_group[g] > 2 || type === 0) continue;                        // collision-only geoms, the MuJoCo floor
  const mat = m.geom_matid[g], c = mat >= 0 ? m.mat_rgba.subarray(4 * mat, 4 * mat + 4) : m.geom_rgba.subarray(4 * g, 4 * g + 4);
  if (c[3] < 0.05) continue;
  let geo;
  if (type === 7) {                                                       // mesh
    const id = m.geom_dataid[g], v0 = m.mesh_vertadr[id], nv = m.mesh_vertnum[id], f0 = m.mesh_faceadr[id], nf = m.mesh_facenum[id];
    geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(m.mesh_vert.subarray(3 * v0, 3 * (v0 + nv))), 3));
    geo.setIndex(Array.from(m.mesh_face.subarray(3 * f0, 3 * (f0 + nf))));
    geo = geo.toNonIndexed();                                              // flat shading for CAD edges
  } else if (type === 6) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  else if (type === 2) geo = new THREE.SphereGeometry(s[0], 24, 16);
  else if (type === 5) geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
  else if (type === 3) geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
  else continue;
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), roughness: 0.55, metalness: 0.05 }));
  mesh.castShadow = mesh.receiveShadow = true; mesh.matrixAutoUpdate = false;
  scene.add(mesh); geoms.push([g, mesh]);
}
const pose = () => {
  for (const [g, mesh] of geoms) {
    const p = d.geom_xpos.subarray(3 * g, 3 * g + 3), r = d.geom_xmat.subarray(9 * g, 9 * g + 9);
    mesh.matrix.set(r[0], r[1], r[2], p[0], r[3], r[4], r[5], p[1], r[6], r[7], r[8], p[2], 0, 0, 0, 1);
  }
};

// ---- target marker + tap ----
const marker = new THREE.Mesh(new THREE.RingGeometry(0.025, 0.04, 40), new THREE.MeshBasicMaterial({ color: '#0071e3' }));
marker.position.z = 0.001; marker.visible = false; scene.add(marker);
let axes = [0, 0, 0], target = null, down = null;
const ray = new THREE.Raycaster(), ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;        // a drag, not a tap
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
  const hit = ray.ray.intersectPlane(ground, new THREE.Vector3());
  if (hit) { target = [hit.x, hit.y]; marker.position.set(hit.x, hit.y, 0.001); marker.visible = true; }
});
const drive = k => { axes = KEYS[k]; target = null; marker.visible = false; };
document.querySelectorAll('[data-k]').forEach(b => b.addEventListener('click', () => drive(b.dataset.k)));
const KB = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', ',': 'sleft', '.': 'sright', ' ': 'stop' };
addEventListener('keydown', e => { if (KB[e.key]) { e.preventDefault(); drive(KB[e.key]); } });
let obs = sim.reset();
$('reset').onclick = () => { obs = sim.reset(); drive('stop'); };
$('push').onclick = () => {                    // a shove like the training pushes (utils/teleop.py DEFAULT_PUSH_RANGE)
  const a = Math.random() * 2 * Math.PI;
  d.qvel[0] += 0.3 * Math.cos(a); d.qvel[1] += 0.3 * Math.sin(a);
};

// ---- 50 Hz control loop (sim time follows the wall clock, at most 4 steps per tick) + render loop ----
let simT = performance.now(), busy = false;
setInterval(async () => {
  if (busy) return;
  busy = true;
  for (let n = 0; n < 4 && simT < performance.now(); n++, simT += sim.STEP_DT * 1000) {
    if (target) axes = goToAxes(...sim.pose(), ...target);
    if (target && axes.every(v => v === 0)) { target = null; marker.visible = false; }
    const c = sim.axesToCommand(...axes);
    obs = sim.step(await policy(obs));
    if (sim.fell) { obs = sim.reset(); drive('stop'); }
    $('status').textContent = `vx ${c[0].toFixed(2)}  vy ${c[1].toFixed(2)}  wz ${c[2].toFixed(2)}`;
  }
  simT = Math.max(simT, performance.now() - 100);                         // drop time we could not keep up with
  busy = false;
}, 10);

const resize = () => {
  renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight;
  camera.fov = camera.aspect < 1 ? 62 : 40;                              // portrait phones: keep the robot and floor in view
  camera.updateProjectionMatrix();
};
addEventListener('resize', resize); resize();
const look = new THREE.Vector3();
renderer.setAnimationLoop(() => {
  pose();
  const [x, y] = sim.pose();
  look.set(x, y, 0.25).sub(controls.target).multiplyScalar(0.08);        // follow the robot, keep the view angle
  controls.target.add(look); camera.position.add(look);
  sun.position.set(x + 1.5, y - 2, 3); sun.target.position.set(x, y, 0);
  controls.update();
  renderer.render(scene, camera);
});
$('loading').style.opacity = 0; setTimeout(() => $('loading').remove(), 400);
