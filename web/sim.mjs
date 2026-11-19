// sim.mjs -- Chhaya in MuJoCo (WebAssembly), step for step as training/envs/mujoco/biped_env_v2.py runs it:
// residual action map, EMA target filter and the BAM servo model every 5 ms substep, the 74-dim observation
// frame with its history, the gait clock and the command limit. Numbers come from assets/robot.json
// (training/src/mujoco/export_web.py). Runs in the browser and in Node (test_sim.mjs).

const ERROR_GAIN = 0.166, MAX_PWM = 0.97;        // STS3215 firmware (actuator/bam_kernel.py)
const FLAGS = { m1: [0, 0, 0, 0], m2: [0, 0, 1, 0], m3: [1, 0, 0, 0], m4: [1, 0, 1, 0], m5: [1, 1, 1, 0], m6: [1, 1, 1, 1] };

// BAM servo (actuator/bam_kernel.py BamKernel.motor + stop_clip), n joints, no backlash.
class Bam {
  constructor(p, n, dt, kp, vin, deadband, torqueScale) {
    Object.assign(this, { p, n, dt, kp, vin, deadband, lim: MAX_PWM * torqueScale });
    const d = p.command_delay / dt;
    this.d0 = Math.floor(d); this.frac = d - this.d0; this.len = this.d0 + 2;
    this.ring = Array.from({ length: this.len }, () => new Float64Array(n));
    this.qs = new Float64Array(n); this.head = 0; this.pending = true;
  }
  reset() { this.pending = true; }
  step(goal, q, dq, ext, inertia, out) {
    const p = this.p, [load, dir, strib, quad] = FLAGS[p.model], ms = p.max_velocity * this.dt;
    if (this.pending) { this.qs.set(q); this.ring.forEach(r => r.set(goal)); this.pending = false; }
    this.head = (this.head + 1) % this.len;
    this.ring[this.head].set(goal);
    const r0 = this.ring[(this.head - this.d0 + 2 * this.len) % this.len];
    const r1 = this.ring[(this.head - this.d0 - 1 + 2 * this.len) % this.len];
    for (let i = 0; i < this.n; i++) {
      const g = (this.d0 === 0 && this.frac === 0) ? goal[i] : (1 - this.frac) * r0[i] + this.frac * r1[i];
      this.qs[i] = Math.min(Math.max(g, this.qs[i] - ms), this.qs[i] + ms);
      let err = this.qs[i] - q[i];
      if (Math.abs(err) <= this.deadband) err = 0;
      let duty = err * this.kp * ERROR_GAIN * p.error_gain_ratio;
      if (p.kd_fw) duty -= dq[i] * p.kd_fw * ERROR_GAIN * p.error_gain_ratio;
      duty = Math.min(Math.max(duty, -this.lim), this.lim);
      const tm = p.kt * (this.vin * duty) / p.R - p.kt ** 2 * dq[i] / p.R;
      // friction budget (bam/model.py compute_frictions)
      const te = ext[i];
      const gb = dir ? Math.abs(te * p.load_friction_external - tm * p.load_friction_motor) : Math.abs(te - tm);
      const gbs = dir ? Math.abs(te * p.load_friction_external_stribeck - tm * p.load_friction_motor_stribeck) : 0;
      const sc = strib ? Math.exp(-(Math.abs(dq[i] / p.dtheta_stribeck) ** p.alpha)) : 0;
      let fl = p.friction_base;
      if (load) fl += dir ? gb : p.load_friction_base * gb;
      if (strib) {
        fl += sc * p.friction_stribeck;
        if (load) fl += dir ? gbs * sc : p.load_friction_stribeck * gb * sc;
        if (load && quad && Math.sign(te) !== Math.sign(tm)) {
          const toM = Math.abs(te) < Math.abs(tm), toE = Math.abs(te) > Math.abs(tm);
          fl += sc * ((toM ? p.load_friction_external_quad * te * te : 0) + (toE ? p.load_friction_motor_quad * tm * tm : 0));
        }
      }
      const budget = fl + p.friction_viscous * Math.abs(dq[i]);
      const stop = (inertia[i] / this.dt) * dq[i] + (tm + te);
      out[i] = tm - Math.sign(stop) * Math.min(Math.abs(stop), budget);
    }
  }
}

const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

export async function createSim(mujoco, load) {          // load(path) -> Promise<Uint8Array>
  const R = JSON.parse(new TextDecoder().decode(await load('assets/robot.json')));
  const xml = new TextDecoder().decode(await load('assets/chhaya.xml'));
  const vfs = new mujoco.MjVFS();
  const files = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map(x => x[1]))];
  (await Promise.all(files.map(load))).forEach((b, i) => vfs.addBuffer(files[i], b));
  const m = mujoco.MjModel.from_xml_string(xml, vfs), d = new mujoco.MjData(m);
  vfs.delete();
  const num = x => Number(x.length !== undefined ? x[0] : x);
  const qadr = R.joints.map(n => num(m.jnt(n).qposadr)), dadr = R.joints.map(n => num(m.jnt(n).dofadr));
  const aq = R.arms.map(n => num(m.jnt(n).qposadr)), ad = R.arms.map(n => num(m.jnt(n).dofadr));
  const act = R.joints.map(n => m.actuator(n).id), aact = R.arms.map(n => m.actuator(n).id);
  const dofs = [...dadr, ...ad], nv = m.nv, NJ = 12, NA = R.arms.length;
  // the env's model edits: the servo model owns friction and rotor inertia, no ctrl clip
  for (const i of dofs) { m.dof_frictionloss[i] = 0; m.dof_damping[i] = 0; m.dof_armature[i] = R.bam.armature; }
  for (const i of [...act, ...aact]) m.actuator_ctrllimited[i] = 0;
  if (R.ground_friction != null) for (let g = 0; g < m.ngeom; g++) m.geom_friction[3 * g] = R.ground_friction;
  m.opt.timestep = R.physics_dt;
  const base = m.body('base_link').id, imuOff = Array.from(m.site('imu_site').pos);
  const STEP_DT = R.physics_dt * R.decimation;
  const legs = new Bam(R.bam, NJ, R.physics_dt, R.kp, R.vin, R.deadband, R.torque_scale);
  const arms = new Bam(R.bam, NA, R.physics_dt, R.kp, R.vin, R.deadband, R.torque_scale);
  const rhs = new Array(dofs.length * nv).fill(0);
  dofs.forEach((j, r) => { rhs[r * nv + j] = 1; });
  const minv = new mujoco.DoubleBuffer(dofs.length * nv);
  const FRAME = 50 + 3 * NA, S = {};

  const rotWB = () => {             // world -> body (R^T of the base quaternion)
    const [w, x, y, z] = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
    return [[1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]];
  };
  const mul = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
  const vImu = () => {
    const X = d.xmat.subarray(9 * base, 9 * base + 9), Rb = [[X[0], X[1], X[2]], [X[3], X[4], X[5]], [X[6], X[7], X[8]]];
    const om = mul(Rb, [d.qvel[3], d.qvel[4], d.qvel[5]]), r = mul(Rb, imuOff);
    return [d.qvel[0] + om[1] * r[2] - om[2] * r[1], d.qvel[1] + om[2] * r[0] - om[0] * r[2], d.qvel[2] + om[0] * r[1] - om[1] * r[0]];
  };
  const grav = () => mul(rotWB(), [0, 0, -1]);
  const enc = 2 * Math.PI / 4096;

  function buildObs() {
    const Rwb = rotWB();
    const lin = mul(Rwb, [S.acc[0], S.acc[1], S.acc[2] + 9.81]);
    S.imu.push([...lin, d.qvel[3], d.qvel[4], d.qvel[5], ...grav()]);
    if (S.imu.length > R.imu_delay + 1) S.imu.shift();
    const f = new Float32Array(FRAME), imu = S.imu[0];
    f.set(imu, 0); f.set(S.cmd, 9);
    for (let i = 0; i < NJ; i++) { f[12 + i] = d.qpos[qadr[i]] - Math.fround(R.home[i]); f[24 + i] = d.qvel[dadr[i]]; f[36 + i] = S.prev[i]; }
    f[48] = Math.sin(S.phase); f[49] = Math.cos(S.phase);
    for (let i = 0; i < NA; i++) {
      f[50 + i] = Math.round(d.qpos[aq[i]] / enc) * enc - R.arm_rest[i];
      f[50 + NA + i] = Math.round(d.qvel[ad[i]] / enc) * enc;
      f[50 + 2 * NA + i] = S.armPrev[i];
    }
    S.hist.push(f); S.hist.shift();
    const o = new Float32Array(FRAME * R.history);
    S.hist.forEach((h, k) => o.set(h, k * FRAME));
    return o;
  }

  function freq(c) {                // CadenceGrid.frequency: trilinear, each axis clamped to the grid
    const G = R.gait, axes = [G.vx, G.vy, G.wz], idx = axes.map((g, k) => {
      if (g.length === 1) return [0, 0];
      const x = clip(c[k], g[0], g[g.length - 1]);
      let i = 0; while (i < g.length - 2 && g[i + 1] <= x) i++;
      return [i, (x - g[i]) / (g[i + 1] - g[i])];
    });
    const [n0, n1, n2] = axes.map(a => a.length);
    let out = 0;
    for (const a of [0, 1]) for (const b of [0, 1]) for (const e of [0, 1]) {
      const w = (a ? idx[0][1] : 1 - idx[0][1]) * (b ? idx[1][1] : 1 - idx[1][1]) * (e ? idx[2][1] : 1 - idx[2][1]);
      const i = Math.min(idx[0][0] + a, n0 - 1), j = Math.min(idx[1][0] + b, n1 - 1), k = Math.min(idx[2][0] + e, n2 - 1);
      out += w * G.freq[(i * n1 + j) * n2 + k];
    }
    return out;
  }

  function setCommand(vx, vy, wz) {   // utils/command_limit.py limit_command
    const r = R.ranges;
    vx = clip(vx, ...r.lin_vel_x); vy = clip(vy, ...r.lin_vel_y); wz = clip(wz, ...r.ang_vel_z);
    if (R.shape === 'ellipse_xy') {
      const bx = vx >= 0 ? r.lin_vel_x[1] : -r.lin_vel_x[0], by = vy >= 0 ? r.lin_vel_y[1] : -r.lin_vel_y[0];
      const s = Math.max(Math.hypot(bx > 0 ? vx / bx : 0, by > 0 ? vy / by : 0), 1);
      vx /= s; vy /= s;
    }
    S.cmd = Float32Array.of(vx, vy, wz);
    return S.cmd;
  }

  function reset(cmd = [0, 0, 0]) {   // BipedEnvV2.reset + the start export_web.py's trace uses
    mujoco.mj_resetData(m, d);
    qadr.forEach((a, i) => { d.qpos[a] = R.home[i]; });
    d.qpos[2] += R.spawn_dz;
    mujoco.mj_forward(m, d);
    S.filt = Float32Array.from(qadr, a => d.qpos[a]);
    S.prev = new Float32Array(NJ); S.armPrev = new Float32Array(NA);
    legs.reset();
    aq.forEach((a, i) => { d.qpos[a] = R.arm_rest[i]; }); ad.forEach(a => { d.qvel[a] = 0; });
    mujoco.mj_forward(m, d);
    arms.reset();
    S.armGoal = Float64Array.from(R.arm_rest); S.armTarget = Float64Array.from(R.arm_rest);
    S.acc = [0, 0, 0]; S.phase = 0; S.cmd = new Float32Array(3); S.fell = false;
    const g = grav();
    S.imu = Array.from({ length: R.imu_delay + 1 }, () => [0, 0, 0, d.qvel[3], d.qvel[4], d.qvel[5], ...g]);
    S.hist = Array.from({ length: R.history }, () => new Float32Array(FRAME));
    buildObs();                                           // the reset observation (discarded, as in the trace)
    setCommand(...cmd);
    S.hist = Array.from({ length: R.history }, () => new Float32Array(FRAME));
    return (S.obs = buildObs());
  }

  const q = new Float64Array(NJ), dq = new Float64Array(NJ), ext = new Float64Array(NJ), ine = new Float64Array(NJ), tau = new Float64Array(NJ);
  const qa = new Float64Array(NA), dqa = new Float64Array(NA), exa = new Float64Array(NA), ina = new Float64Array(NA), taua = new Float64Array(NA);

  function step(action) {             // action [12 legs | NA arms] in [-1, 1] -> next observation
    const a = Float32Array.from(action, x => clip(x, -1, 1));
    const tgt = new Float64Array(NJ);
    for (let i = 0; i < NJ; i++) {
      const s = R.scale_neg && a[i] < 0 ? R.scale_neg[i] : R.scale[i];
      tgt[i] = clip(R.home[i] + s * a[i], R.jlim_min[i], R.jlim_max[i]);
    }
    for (let i = 0; i < NA; i++) S.armTarget[i] = clip(R.arm_rest[i] + R.arm_scale[i] * a[NJ + i], R.arm_lim_min[i], R.arm_lim_max[i]);
    const c = S.cmd, moving = Math.hypot(c[0], c[1]) >= R.gait.stand || Math.abs(c[2]) >= R.gait.stand;
    S.phase = (S.phase + STEP_DT * 2 * Math.PI * (moving ? freq(c) : 0)) % (2 * Math.PI);
    let v0;
    for (let k = 0; k < R.decimation; k++) {
      for (let i = 0; i < NJ; i++) S.filt[i] = R.alpha * tgt[i] + (1 - R.alpha) * S.filt[i];
      mujoco.mj_solveM(m, d, minv, rhs);
      const Mi = minv.GetView();
      dofs.forEach((j, r) => {
        const v = 1 / Mi[r * nv + j], e = -d.qfrc_bias[j] + d.qfrc_constraint[j];
        if (r < NJ) { q[r] = d.qpos[qadr[r]]; dq[r] = d.qvel[j]; ext[r] = e; ine[r] = v; }
        else { const i = r - NJ; qa[i] = d.qpos[aq[i]]; dqa[i] = d.qvel[j]; exa[i] = e; ina[i] = v; }
      });
      legs.step(S.filt, q, dq, ext, ine, tau);
      for (let i = 0; i < NA; i++) S.armGoal[i] = R.alpha * S.armTarget[i] + (1 - R.alpha) * S.armGoal[i];
      arms.step(S.armGoal, qa, dqa, exa, ina, taua);
      act.forEach((id, i) => { d.ctrl[id] = tau[i]; }); aact.forEach((id, i) => { d.ctrl[id] = taua[i]; });
      if (k === 0) v0 = vImu();
      mujoco.mj_step(m, d);
    }
    const v1 = vImu();
    S.acc = [0, 1, 2].map(i => (v1[i] - v0[i]) / STEP_DT);
    S.prev = a.slice(0, NJ); S.armPrev = a.slice(NJ);
    S.fell = Math.acos(clip(-grav()[2], -1, 1)) > R.tilt_limit;
    return (S.obs = buildObs());
  }

  function axesToCommand(fwd, lat, turn, level = 0.75) {   // utils/teleop.py axes_to_command
    const r = R.ranges, edge = (v, [lo, hi]) => (v >= 0 ? v * hi : -v * lo), k = x => clip(x, -1, 1) * level;
    return setCommand(edge(k(fwd), r.lin_vel_x), edge(k(lat), r.lin_vel_y), edge(k(turn), r.ang_vel_z));
  }

  function pose() {
    const [w, x, y, z] = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
    return [d.qpos[0], d.qpos[1], Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))];
  }

  const norm = o => Float32Array.from(o, (x, i) => (x - R.obs_mean[i]) / R.obs_std[i]);
  return { m, d, R, reset, step, setCommand, axesToCommand, pose, norm, get fell() { return S.fell; }, get obs() { return S.obs; }, STEP_DT };
}

export function goToAxes(x, y, yaw, tx, ty, arrive = 0.08) {   // utils/teleop.py go_to_axes
  if (Math.hypot(tx - x, ty - y) < arrive) return [0, 0, 0];
  const head = ((Math.atan2(ty - y, tx - x) - yaw + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  return [Math.max(0, Math.cos(head)) ** 2, 0, clip(head / (Math.PI / 4), -1, 1)];
}
