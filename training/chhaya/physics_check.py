"""
Physics readiness checks for the 20-joint (legs + arms) MuJoCo model, BEFORE policy training.

Every joint is driven the way envs/mujoco/biped_env_v2.py drives the legs: the BAM STS3215
model (actuator/bam_kernel.py) turns a target angle into a joint torque each 200 Hz substep,
with the env's per-dof setup (frictionloss 0, damping 0, armature from the fit, stall cap
2.94 N*m, firmware P 32, 1-count dead zone). No policy: the legs hold the crouched home pose,
so these tests measure the model itself, not a controller.

    python -m chhaya.physics_check            (run from training/; writes ../docs/physics)

Tests
  T0  model sanity: compiles, finite, inertias physical, joint signs as documented
  T1  stand: settle from a drop and hold home for 5 s (vs the legs-only robot_v2 baseline)
  T2  static holding: worst-case arm poses (T-pose, arms forward) -- arm torque vs the stall cap
  T3  arm tracking: per-joint sinusoids -- tracking error, lag, saturation
  T4  arm-motion disturbance: swing / wave / fast random arms while the legs just hold home
  T5  leg clearance: random leg poses with the arms at rest -> arm/leg contacts (contacts during
      T1-T4 are reported inside those tests)
  T6  robustness: half the timestep, and with backlash hinges on all 20 joints
  T7  torque off: servos limp -> no numerical blow-up (the gearbox friction may hold the stance)
  T8  fall: shoved over with servos on -> impact, no NaN, peak joint torques
  T9  speed: raw MuJoCo steps/s vs the legs-only model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from actuator.bam_kernel import BamServo, apply_stall_cap, load_params   # noqa: E402
from chhaya.model import ARM_JOINTS, LEG_JOINTS                      # noqa: E402
from chhaya import motions                                           # noqa: E402

PARAMS = "actuator/params/sts3215_12v/m6.json"
DT = 0.005
DECIM = 4
HOME_DEG = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -10.0, "knee": -26.0, "ankle_roll": 0.0, "ankle_pitch": -16.0}
XML_CHHAYA = REPO / "envs/assets/robotV2/chhaya.xml"
XML_LEGS = REPO / "envs/assets/robotV2/robot_v2.xml"


def leg_home():
    out = {}
    for j in LEG_JOINTS:
        key = j[2:].replace("_joint", "")
        out[j] = math.radians(HOME_DEG[key])
    return out


class Rig:
    """MuJoCo model + BAM servos on the listed joints, set up like biped_env_v2."""

    def __init__(self, xml, joints, dt=DT, backlash_deg=0.0):
        xml = str(xml)
        if backlash_deg > 0:
            from actuator.backlash import add_backlash
            out = os.path.join(os.path.dirname(xml), "_physcheck.backlash.xml")
            xml = str(add_backlash(xml, out, joints, backlash_deg))
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.m.opt.timestep = dt
        self.d = mujoco.MjData(self.m)
        self.joints = list(joints)
        n = len(self.joints)
        self.qpos_ids = np.array([self.m.joint(j).qposadr[0] for j in self.joints])
        self.qvel_ids = np.array([self.m.joint(j).dofadr[0] for j in self.joints])
        self.act_ids = np.array([self.m.actuator(j).id for j in self.joints])
        p = apply_stall_cap(load_params(REPO / PARAMS), 2.94)
        self.servo = BamServo(p, 1, n, dt, kp=32.0, vin=12.0, backlash=False, dtype=torch.float64)
        self.servo.kernel.deadband[:] = 1 * (2 * math.pi / 4096)
        self.m.dof_frictionloss[self.qvel_ids] = 0.0
        self.m.dof_damping[self.qvel_ids] = 0.0
        self.m.dof_armature[self.qvel_ids] = p.armature
        self.m.actuator_ctrllimited[self.act_ids] = 0
        self.rhs = np.zeros((n, self.m.nv))
        self.rhs[np.arange(n), self.qvel_ids] = 1.0
        self.minv = np.zeros_like(self.rhs)
        self.enable = 1.0
        self.goal = np.zeros(n)
        self.tau = np.zeros(n)
        self.base = self.m.body("base_link").id

    def set_goal(self, pose: dict):
        for i, j in enumerate(self.joints):
            if j in pose:
                self.goal[i] = pose[j]

    def reset(self, pose: dict, z=None):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qpos_ids] = [pose.get(j, 0.0) for j in self.joints]
        self.set_goal(pose)
        mujoco.mj_forward(self.m, self.d)
        if z is None:   # lowest foot point 5 mm above the floor
            z = self.d.qpos[2] - self.lowest_foot_z() + 0.005
        self.d.qpos[2] = z
        mujoco.mj_forward(self.m, self.d)
        self.servo.reset()

    def lowest_foot_z(self):
        zs = []
        for g in range(self.m.ngeom):
            b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) or ""
            if "foot" not in b or self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            w = v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            zs.append(w[:, 2].min())
        return min(zs)

    def substep(self):
        q = self.d.qpos[self.qpos_ids].copy()
        dq = self.d.qvel[self.qvel_ids].copy()
        tau_ext = -self.d.qfrc_bias[self.qvel_ids] + self.d.qfrc_constraint[self.qvel_ids]
        mujoco.mj_solveM(self.m, self.d, self.minv, self.rhs)
        inv = self.minv[np.arange(len(self.joints)), self.qvel_ids]
        t = lambda x: torch.from_numpy(np.asarray(x, np.float64)).unsqueeze(0)
        tau = self.servo.step(t(self.goal), t(q), t(dq), t(tau_ext), t(1.0 / inv), torque_enable=self.enable)
        self.tau = tau[0].numpy()
        self.d.ctrl[self.act_ids] = self.tau
        mujoco.mj_step(self.m, self.d)

    # -- measurements --------------------------------------------------------------------
    def tilt_deg(self):
        R = self.d.xmat[self.base].reshape(3, 3)
        return math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))

    def height(self):
        return float(self.d.xpos[self.base][2])

    def finite(self):
        return bool(np.all(np.isfinite(self.d.qpos)) and np.all(np.isfinite(self.d.qvel)))

    def saturated(self):
        return self.servo.duty_saturated[0].numpy()

    def contacts(self):
        """(pairs between different 'groups', max penetration) -- groups: floor, legs, base/chest, each arm."""
        def group(b):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
            if n == "world":
                return "floor"
            for s in "lr":
                if n.startswith(s + "_") and any(k in n for k in ("shoulder", "upper_arm", "elbow", "forearm")):
                    return s + "_arm"
            if n in ("base_link", "chest_link"):
                return "torso"
            return "legs"
        pairs, pen = {}, 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            ga, gb = group(self.m.geom_bodyid[c.geom1]), group(self.m.geom_bodyid[c.geom2])
            if "arm" in ga + gb:
                key = "-".join(sorted((ga, gb)))
                pairs[key] = pairs.get(key, 0) + 1
                pen = min(pen, c.dist)
        return pairs, pen


def run(rig, seconds, target_fn=None, record=None):
    """target_fn(t) -> pose dict, called at 50 Hz. Returns time series."""
    n_ctrl = int(round(seconds / (DT * DECIM))) if rig.m.opt.timestep == DT else int(round(seconds / (rig.m.opt.timestep * DECIM)))
    sub = DECIM if rig.m.opt.timestep == DT else int(round(DT * DECIM / rig.m.opt.timestep))
    ts = {"t": [], "tilt": [], "z": [], "tau": [], "q": [], "goal": [], "sat": [], "contacts": [], "pen": []}
    t = 0.0
    for k in range(n_ctrl):
        if target_fn is not None:
            rig.set_goal(target_fn(t))
        for _ in range(sub):
            rig.substep()
            if not rig.finite():
                ts["nan_at"] = t
                return {k2: np.array(v) if k2 != "contacts" else v for k2, v in ts.items()}
        t += rig.m.opt.timestep * sub
        pairs, pen = rig.contacts()
        ts["t"].append(t); ts["tilt"].append(rig.tilt_deg()); ts["z"].append(rig.height())
        ts["tau"].append(rig.tau.copy()); ts["q"].append(rig.d.qpos[rig.qpos_ids].copy())
        ts["goal"].append(rig.goal.copy()); ts["sat"].append(rig.saturated().copy())
        ts["contacts"].append(pairs); ts["pen"].append(pen)
    return {k2: (np.array(v) if k2 != "contacts" else v) for k2, v in ts.items()}


def summarize_stand(ts, joints, t0=1.0):
    sel = ts["t"] >= t0
    tau = ts["tau"][sel]
    out = {"fell": bool(ts["tilt"].max() > 30 or ts["z"][-1] < 0.18), "nan": "nan_at" in ts,
           "tilt_final_deg": round(float(ts["tilt"][-1]), 2), "tilt_max_after_settle_deg": round(float(ts["tilt"][sel].max()), 2),
           "height_final_m": round(float(ts["z"][-1]), 4), "height_drift_mm": round(float((ts["z"][sel].max() - ts["z"][sel].min()) * 1000), 2)}
    out["torque_rms"] = {j: round(float(np.sqrt((tau[:, i] ** 2).mean())), 3) for i, j in enumerate(joints)}
    out["torque_peak"] = {j: round(float(np.abs(tau[:, i]).max()), 3) for i, j in enumerate(joints)}
    out["saturation_pct"] = {j: round(float(ts["sat"][sel][:, i].mean() * 100), 1) for i, j in enumerate(joints)}
    return out


# ------------------------------------------------------------------------------------------
def T0_sanity():
    m = mujoco.MjModel.from_xml_path(str(XML_CHHAYA))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    res = {"compiles": True, "nq": m.nq, "nv": m.nv, "nu": m.nu, "mass_kg": round(float(m.body_subtreemass[1]), 4),
           "finite": bool(np.all(np.isfinite(d.qacc)))}
    bad = []
    for b in range(1, m.nbody):
        I = m.body_inertia[b]
        if m.body_mass[b] <= 0 or np.any(I <= 0) or I[0] + I[1] < I[2] - 1e-12 or I[1] + I[2] < I[0] - 1e-12 or I[0] + I[2] < I[1] - 1e-12:
            bad.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
    res["bodies_with_unphysical_inertia"] = bad
    # sign conventions (kinematic): move each joint +0.3 rad and see where the forearm goes
    signs = {}
    fl = {s: m.body(f"{s}_forearm_link").id for s in "lr"}
    for j in ARM_JOINTS:
        s = j[0]
        mujoco.mj_resetData(m, d)
        if j.endswith("arm_yaw"):
            d.qpos[m.joint(f"{s}_elbow").qposadr[0]] = -1.2       # bend the elbow so yaw has a visible effect
        mujoco.mj_forward(m, d)
        p0 = d.xipos[fl[s]].copy()
        d.qpos[m.joint(j).qposadr[0]] += 0.3
        mujoco.mj_forward(m, d)
        dp = d.xipos[fl[s]] - p0
        out = dp[1] * (1 if s == "l" else -1)
        if j.endswith("shoulder_pitch"):
            ok, what = dp[0] < 0, "hand moves backward"
        elif j.endswith("shoulder_roll"):
            ok, what = out > 0, "hand moves outward"
        elif j.endswith("arm_yaw"):
            ok, what = out > 0, "bent forearm swings outward"
        else:
            ok, what = dp[0] < 0 or dp[2] < 0, "forearm extends (positive = straighten)"
        signs[j] = {"expected": what, "ok": bool(ok), "dpos_mm": [round(float(x) * 1000, 1) for x in dp]}
    res["sign_conventions"] = signs
    return res


def T1_stand(out):
    res = {}
    for name, xml, joints, pose in (("legs_only_baseline", XML_LEGS, LEG_JOINTS, leg_home()),
                                    ("with_arms", XML_CHHAYA, LEG_JOINTS + ARM_JOINTS, {**leg_home(), **motions.REST})):
        rig = Rig(xml, joints)
        rig.reset(pose)
        ts = run(rig, 6.0)
        s = summarize_stand(ts, joints)
        s["contacts_at_rest"] = ts["contacts"][-1]
        res[name] = s
        out[name] = ts
    return res


def stand_rig():
    rig = Rig(XML_CHHAYA, LEG_JOINTS + ARM_JOINTS)
    rig.reset({**leg_home(), **motions.REST})
    run(rig, 1.5)
    return rig


def T2_holding():
    res = {}
    for pose_name in ("t_pose", "hands_forward", "hands_up"):
        rig = stand_rig()
        a, b = motions.REST, motions.POSES[pose_name]
        ts = run(rig, 4.0, lambda t: {**leg_home(), **motions.blend(a, b, t / 1.5)})
        sel = ts["t"] > 2.5
        idx = [rig.joints.index(j) for j in ARM_JOINTS]
        err = np.abs(ts["q"][sel][:, idx] - ts["goal"][sel][:, idx])
        res[pose_name] = {
            "arm_torque_hold": {j: round(float(np.abs(ts["tau"][sel][:, i]).mean()), 3) for j, i in zip(ARM_JOINTS, idx)},
            "worst_arm_torque_pct_of_stall": round(float(np.abs(ts["tau"][sel][:, idx]).mean(0).max() / 2.94 * 100), 1),
            "arm_tracking_err_deg_max": round(float(np.degrees(err.max())), 2),
            "robot_tilt_deg": round(float(ts["tilt"][-1]), 2), "fell": bool(ts["tilt"].max() > 30),
            "leg_torque_peak": round(float(np.abs(ts["tau"][sel][:, :12]).max()), 3),
        }
    return res


def T3_tracking():
    res = {}
    for j in ARM_JOINTS[::2]:            # left arm (the right is the mirror)
        rig = stand_rig()
        i = rig.joints.index(j)
        base = dict(motions.REST)
        if j.endswith("arm_yaw"):
            base["l_elbow"] = -1.2
        lo, hi = {"l_shoulder_pitch": (-1.2, 0.4), "l_shoulder_roll": (0.1, 1.3), "l_arm_yaw": (-0.7, 0.7), "l_elbow": (-1.6, -0.1)}[j]
        mid, amp = (lo + hi) / 2, (hi - lo) / 2
        rig.set_goal(base)
        run(rig, 1.0, lambda t: {**leg_home(), **base, j: base[j] + (mid - base[j]) * min(1, t)})
        for f in (0.5, 1.5):
            ts = run(rig, 3.0 / f + 1.0, lambda t: {**leg_home(), **base, j: mid + amp * math.sin(2 * math.pi * f * t)})
            sel = ts["t"] > 1.0
            q, g = ts["q"][sel][:, i], ts["goal"][sel][:, i]
            # lag by cross-correlation (50 Hz samples)
            qq, gg = q - q.mean(), g - g.mean()
            lags = range(0, 25)
            corr = [np.dot(gg[:len(gg) - L], qq[L:]) for L in lags]
            lag = lags[int(np.argmax(corr))] * DT * DECIM
            res[f"{j}@{f}Hz"] = {"amp_rad": round(amp, 2), "err_rms_deg": round(float(np.degrees(np.sqrt(((q - g) ** 2).mean()))), 2),
                                  "lag_ms": round(lag * 1000, 0), "sat_pct": round(float(ts["sat"][sel][:, i].mean() * 100), 1),
                                  "torque_peak": round(float(np.abs(ts["tau"][sel][:, i]).max()), 3)}
    return res


def T4_disturbance(out):
    res = {}
    rnd = motions.RandomArms(rng=np.random.default_rng(1), bandwidth_hz=1.2, amplitude=0.8)
    cases = {
        "arm_swing_1.5Hz": lambda t: motions.arm_swing((1.5 * t) % 1.0, amplitude=0.5),
        "wave": lambda t: motions.wave(t),
        "raise_and_lower": motions.GESTURES["raise_and_lower"],
        "random_fast": lambda t: rnd.step(DT * DECIM),
    }
    for name, fn in cases.items():
        rig = stand_rig()
        ts = run(rig, 8.0, lambda t: {**leg_home(), **fn(t)})
        sel = ts["t"] > 0.5
        res[name] = {"fell": bool(ts["tilt"].max() > 30 or ts["z"][-1] < 0.18), "nan": "nan_at" in ts,
                     "tilt_peak_deg": round(float(ts["tilt"][sel].max()), 2),
                     "leg_torque_peak": round(float(np.abs(ts["tau"][sel][:, :12]).max()), 3),
                     "leg_sat_pct": round(float(ts["sat"][sel][:, :12].mean() * 100), 2),
                     "arm_sat_pct": round(float(ts["sat"][sel][:, 12:].mean() * 100), 2),
                     "arm_contact_steps_pct": round(100 * float(np.mean([bool(c) for c in ts["contacts"]])), 1),
                     "contact_pairs_seen": sorted({k for c in ts["contacts"] for k in c}),
                     "max_penetration_mm": round(float(-ts["pen"].min() * 1000), 2)}
        out["T4_" + name] = ts
    return res


def T6_robustness():
    res = {}
    for name, kw in (("dt_0.0025", {"dt": 0.0025}), ("backlash_1.5deg_all_20", {"backlash_deg": 1.5})):
        rig = Rig(XML_CHHAYA, LEG_JOINTS + ARM_JOINTS, **kw)
        rig.reset({**leg_home(), **motions.REST})
        ts = run(rig, 4.0, lambda t: {**leg_home(), **motions.arm_swing((1.0 * t) % 1.0, 0.4)})
        res[name] = {"nan": "nan_at" in ts, "fell": bool(ts["tilt"].max() > 30),
                     "tilt_final_deg": round(float(ts["tilt"][-1]), 2), "height_final_m": round(float(ts["z"][-1]), 4)}
    return res


def T7_torque_off():
    rig = stand_rig()
    rig.enable = 0.0
    ts = run(rig, 3.0)
    vmax = float(np.abs(rig.d.qvel).max()) if rig.finite() else float("nan")
    return {"nan": "nan_at" in ts, "collapsed_height_m": round(float(ts["z"][-1]), 3),
            "max_abs_qvel_end": round(vmax, 3), "stays_finite": rig.finite()}


def T5_leg_clearance(n=2000):
    """Static: random leg poses (gait ranges, and the full soft limits) with the arms at rest --
    how often an arm collision geom touches a leg."""
    m = mujoco.MjModel.from_xml_path(str(XML_CHHAYA))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    full = {"hip_yaw": (-0.785, 0.785), "hip_roll": (-0.436, 0.785), "hip_pitch": (-1.571, 0.524), "knee": (-2.094, 0.087),
            "ankle_roll": (-1.047, 1.047), "ankle_pitch": (-0.611, 0.873)}
    gait = {"hip_yaw": (-0.45, 0.45), "hip_roll": (-0.25, 0.45), "hip_pitch": (-1.0, 0.4), "knee": (-1.6, 0.0),
            "ankle_roll": (-0.35, 0.35), "ankle_pitch": (-0.5, 0.6)}

    def arm(g):
        nme = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "world"
        return any(k in nme for k in ("shoulder", "upper_arm", "elbow", "forearm")), nme == "world"
    res = {}
    for lname, L in (("gait_ranges", gait), ("full_soft_limits", full)):
        cnt, worst = 0, 0.0
        for _ in range(n):
            mujoco.mj_resetData(m, d)
            d.qpos[2] = 1.0
            for j in LEG_JOINTS:
                d.qpos[m.joint(j).qposadr[0]] = rng.uniform(*L[j[2:].replace("_joint", "")])
            for j, v in motions.REST.items():
                d.qpos[m.joint(j).qposadr[0]] = v
            mujoco.mj_forward(m, d)
            hit = False
            for i in range(d.ncon):
                c = d.contact[i]
                (a1, w1), (a2, w2) = arm(c.geom1), arm(c.geom2)
                if (a1 or a2) and not (w1 or w2):
                    hit, worst = True, min(worst, c.dist)
            cnt += hit
        res[lname] = {"poses_with_arm_leg_contact_pct": round(100 * cnt / n, 1), "worst_overlap_mm": round(-worst * 1000, 1)}
    return res


def T9_speed():
    import time
    out = {}
    for name, xml, joints, pose in (("legs_only", XML_LEGS, LEG_JOINTS, leg_home()),
                                    ("with_arms", XML_CHHAYA, LEG_JOINTS + ARM_JOINTS, {**leg_home(), **motions.REST})):
        rig = Rig(xml, joints)
        rig.reset(pose)
        t, k = time.time(), 0
        while time.time() - t < 3:
            mujoco.mj_step(rig.m, rig.d)
            k += 1
        out[name] = {"mj_step_per_s": int(k / 3), "ngeom": int(rig.m.ngeom), "nv": int(rig.m.nv)}
    return out


def T8_fall():
    base = Rig(XML_LEGS, LEG_JOINTS)
    base.reset(leg_home())
    run(base, 1.5)
    base.d.qvel[0] += 0.8
    base.d.qvel[1] += 0.9
    tsb = run(base, 3.0)
    rig = stand_rig()
    # shove: 1.2 m/s sideways-forward at the base
    rig.d.qvel[0] += 0.8
    rig.d.qvel[1] += 0.9
    ts = run(rig, 3.0, lambda t: {**leg_home(), **motions.REST})
    return {"nan": "nan_at" in ts, "fell": bool(ts["tilt"].max() > 30), "tilt_peak_deg": round(float(ts["tilt"].max()), 1),
            "arm_torque_peak": round(float(np.abs(ts["tau"][:, 12:]).max()), 3),
            "leg_torque_peak": round(float(np.abs(ts["tau"][:, :12]).max()), 3),
            "max_arm_penetration_mm": round(float(-ts["pen"].min() * 1000), 2),
            "legs_only_baseline_peak_tilt_deg": round(float(tsb["tilt"].max()), 1),
            "contact_pairs_seen": sorted({k for c in ts["contacts"] for k in c})}


def plots(series, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, ts in series.items():
        if name.startswith("T4_") or name in ("legs_only_baseline", "with_arms"):
            ax[0].plot(ts["t"], ts["tilt"], label=name)
            ax[1].plot(ts["t"], ts["z"], label=name)
    for name in ("legs_only_baseline", "with_arms"):
        ts = series[name]
        ax[2].plot(ts["t"], np.abs(ts["tau"][:, :12]).max(1), label=name + " (max |leg tau|)")
    ax[0].set_title("base tilt [deg]"); ax[1].set_title("base height [m]"); ax[2].set_title("max |leg torque| [N*m]")
    for a in ax:
        a.grid(True, alpha=0.3); a.set_xlabel("t [s]")
    ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "physics_overview.png", dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO.parent / "docs" / "physics")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    series = {}
    report = {}
    for name, fn in (("T0_sanity", T0_sanity), ("T1_stand", lambda: T1_stand(series)), ("T2_holding", T2_holding),
                     ("T3_tracking", T3_tracking), ("T4_disturbance", lambda: T4_disturbance(series)),
                     ("T5_leg_clearance", T5_leg_clearance), ("T6_robustness", T6_robustness),
                     ("T7_torque_off", T7_torque_off), ("T8_fall", T8_fall), ("T9_speed", T9_speed)):
        print(f"== {name}", flush=True)
        report[name] = fn()
        print(json.dumps(report[name], indent=1)[:2500], flush=True)
    (a.out / "physics_report.json").write_text(json.dumps(report, indent=2))
    plots(series, a.out)


if __name__ == "__main__":
    main()
