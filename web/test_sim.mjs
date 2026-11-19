// test_sim.mjs -- check web/sim.mjs against the Python MuJoCo env, then let the policy walk.
//   node web/test_sim.mjs <dir with node_modules/@mujoco/mujoco and node_modules/onnxruntime-web>
// 1. open loop: the actions of assets/trace.json (export_web.py --trace) must give the same observations;
// 2. closed loop: the policy walks forward and through tapped spots without falling.
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSim, goToAxes } from './sim.mjs';

const web = dirname(fileURLToPath(import.meta.url)), mods = join(process.argv[2], 'node_modules');
const mujoco = await (await import(pathToFileURL(join(mods, '@mujoco/mujoco/mujoco.js')))).default();
const ort = await import(pathToFileURL(join(mods, 'onnxruntime-web/dist/ort.node.min.mjs')));
const sim = await createSim(mujoco, async p => new Uint8Array(await readFile(join(web, p))));
const session = await ort.InferenceSession.create(join(web, 'assets/policy.onnx'));
const policy = async o => Array.from((await session.run({ obs: new ort.Tensor('float32', sim.norm(o), [1, o.length]) })).action.data, x => Math.min(Math.max(x, -1), 1));

// 1. open loop
const T = JSON.parse(await readFile(join(web, 'assets/trace.json'), 'utf8'));
let o = sim.reset(T.cmd), F = T.rows[0].obs.length, worst = [];
for (let i = 0; i < T.rows.length; i++) {
  const ref = T.rows[i].obs, mine = o.subarray(o.length - F);
  worst.push(Math.max(...ref.map((v, k) => Math.abs(v - mine[k]))));
  o = sim.step(T.rows[i].act);
}
const at = n => Math.max(...worst.slice(0, n)).toExponential(1);
// exact until the first foot contact (step 4); after that the MuJoCo versions (Python vs WASM) resolve contacts
// differently and the chaotic gait drifts apart, so only the closed-loop walk below is a criterion
console.log(`open loop, max |obs - python|: steps 0-3 ${at(4)} (must be < 1e-5), 10 ${at(10)}, ${worst.length} ${at(worst.length)}`);
const py = (await policy(sim.reset(T.cmd))).map((v, k) => Math.abs(v - T.rows[0].act[k]));
console.log(`first action vs python: max diff ${Math.max(...py).toExponential(1)}`);

// 2. closed loop: walk forward 10 s, then a square of tapped spots
o = sim.reset([0.15, 0, 0]);
for (let i = 0; i < 500 && !sim.fell; i++) o = sim.step(await policy(o));
const [x] = sim.pose();
console.log(`forward 10 s at 0.15 m/s: x = ${x.toFixed(2)} m, fell = ${sim.fell}`);
let ok = x > 0.8 && !sim.fell && Math.max(...worst.slice(0, 4)) < 1e-5;
o = sim.reset();
for (const t of [[0.6, 0], [0.6, 0.5], [0, 0.5], [0, 0]]) {
  let ax, i = 0;
  for (; i < 2000 && !sim.fell; i++) {
    ax = goToAxes(...sim.pose(), ...t);
    sim.axesToCommand(...ax);
    if (ax.every(v => v === 0)) break;
    o = sim.step(await policy(o));
  }
  const reached = ax.every(v => v === 0) && !sim.fell;
  console.log(`tap ${t}: reached = ${reached} after ${(i * sim.STEP_DT).toFixed(1)} s`);
  ok &&= reached;
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
