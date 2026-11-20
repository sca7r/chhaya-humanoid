"""Can the STS3215 servos carry Chhaya? Joint torques in MuJoCo against the servo's limits.

    python -m chhaya.servo_load [--out docs_dir]

Three load cases, for Chhaya (config/chhaya.yaml):
  1. standing on both feet in the home pose (measured in the simulation: zero action, 3 s);
  2. standing on ONE foot in the same pose (quasi-static: for each stance-leg joint, the gravity moment of
     everything on the far side of the joint; walking loads each leg like this once per step);
  3. arms held straight out (forward and sideways), static gravity moment at the shoulder and elbow.
Limits: stall torque of the fitted BAM model (the absolute ceiling at 12 V), and a continuous limit of
stall / 3 (a common rule for hobby-class servos; Feetech publishes no thermal rating).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

STALL = 2.94          # N m, STS3215 at 12 V (actuator/params/sts3215_12v); replaced per config in main()
CONT = STALL / 3.0    # N m, continuous-duty guide


def stall_from_config(cfg_name):
    """(leg stall, arm stall, description) of the servo model a config uses, at its supply voltage."""
    import yaml
    from actuator.bam_kernel import apply_stall_cap, load_params
    act = yaml.safe_load(open(os.path.join(ROOT, "config", cfg_name)))["train"]["env_config"]["actuator"]
    vin = float(act.get("vin", 12.0))
    out = []
    for key in ("params_file", "arms_params_file"):
        p = apply_stall_cap(load_params(act.get(key) or act["params_file"]), act.get("stall_torque_cap_nm"))
        out.append((p.stall_torque * vin / p.v_nominal, p.actuator))
    return out[0][0], out[1][0], f"legs {out[0][1]}, arms {out[1][1]}, supply {vin:g} V"


def make_env(cfg_name):
    from utils import Config
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    cfg = Config(os.path.join(ROOT, "config", cfg_name))
    ec = cfg.train.env_config
    return BipedEnvV2(history_size=5, use_history=True, env_config=ec, seed=0)


def joint_names(m):
    return [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]


def stand(env, seconds=3.0):
    env.reset(seed=0)
    env.set_command(0.0, 0.0, 0.0)
    n = env.action_space.shape[0]
    m, d = env.model, env.data
    act_joint = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
    rec = []
    for k in range(int(seconds * 50)):
        _, _, term, _, _ = env.step(np.zeros(n, np.float32))
        assert not term, "fell while standing"
        if k >= 50:
            rec.append(np.abs(d.actuator_force.copy()))
    rec = np.array(rec)
    return dict(zip(act_joint, rec.mean(0))), dict(zip(act_joint, rec.max(0))), d.qpos.copy()


def single_support(m, qpos, stance="l"):
    """Static joint torque of the stance leg with the whole robot balanced on that foot."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    g = np.array([0, 0, -9.81])
    out = {}
    for j in range(m.njnt):
        name = m.joint(j).name
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not name.startswith(stance + "_"):
            continue
        if name.endswith("_lash") or any(k in name for k in ("shoulder", "arm", "elbow")):
            continue
        child = m.jnt_bodyid[j]
        sub = set()
        for b in range(m.nbody):
            p = b
            while p != 0:
                if p == child:
                    sub.add(b)
                    break
                p = m.body_parentid[p]
        axis = d.xaxis[j]
        anchor = d.xanchor[j]
        tau = 0.0
        for b in range(1, m.nbody):
            if b in sub:
                continue
            tau += axis @ np.cross(d.xipos[b] - anchor, m.body_mass[b] * g)
        out[name] = abs(float(tau))
    return out


def arm_hold(m, qpos):
    """Static shoulder/elbow torque with the left arm straight out forward and sideways (worst reach)."""
    res = {}
    names = joint_names(m)
    if "l_shoulder_pitch" not in names:
        return res
    for label, pose in (("arm forward", {"l_shoulder_pitch": -1.5708, "l_elbow": 0.0, "l_shoulder_roll": 0.05}),
                        ("arm sideways", {"l_shoulder_roll": 1.5708, "l_elbow": 0.0})):
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        for n, v in pose.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for n in ("l_shoulder_pitch", "l_shoulder_roll", "l_arm_yaw", "l_elbow"):
            res.setdefault(label, {})[n] = abs(float(d.qfrc_bias[m.jnt_dofadr[m.joint(n).id]]))
    return res


def analyse(cfg_name):
    env = make_env(cfg_name)
    mean, peak, qpos = stand(env)
    m = env.model
    mass = float(m.body_mass.sum())
    ss = single_support(m, qpos)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    thigh = float(np.linalg.norm(d.xanchor[m.joint("l_hip_pitch_joint").id] - d.xanchor[m.joint("l_knee_joint").id]))
    kb = m.jnt_bodyid[m.joint("l_knee_joint").id]

    def below(b):
        while b != 0:
            if b == kb:
                return True
            b = m.body_parentid[b]
        return False
    m_low = float(sum(m.body_mass[b] for b in range(m.nbody) if below(b)))
    squat = {str(2 * th): {"one_leg": (mass - m_low) * 9.81 * thigh * np.sin(np.radians(th)),
                           "two_legs": (mass - 2 * m_low) / 2 * 9.81 * thigh * np.sin(np.radians(th))} for th in (20, 30, 45, 60)}
    return {"mass_kg": round(mass, 3), "squat_knee": squat, "stand_mean": mean, "stand_peak": peak, "single_support": ss, "arm_hold": arm_hold(m, qpos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="chhaya.yaml", help="config in config/ (chhaya_lite.yaml for the hobby servos)")
    a = ap.parse_args()
    global STALL, CONT
    STALL, arm_stall, desc = stall_from_config(a.config)
    CONT = STALL / 3.0
    res = {"chhaya": analyse(a.config)}
    ch = res["chhaya"]
    lines = [f"{a.config}: {ch['mass_kg']} kg. Servos: {desc}. "
             f"Leg stall {STALL:.2f} N m (continuous guide {CONT:.2f}), arm stall {arm_stall:.2f} N m.", ""]
    lines += ["| Joint (left leg) | Stand, both feet | One foot, Chhaya | % of stall | % of continuous |", "|---|---|---|---|---|"]
    worst = 0.0
    for n, v in ch["single_support"].items():
        worst = max(worst, v)
        lines.append(f"| {n} | {ch['stand_mean'].get(n, float('nan')):.2f} | {v:.2f} | {100 * v / STALL:.0f} % | {100 * v / CONT:.0f} % |")
    lines += ["", "| Knee bend | Knee torque, both legs | Knee torque, one leg | % of stall (one leg) |", "|---|---|---|---|"]
    for ang, v in ch["squat_knee"].items():
        lines.append(f"| {ang} deg | {v['two_legs']:.2f} | {v['one_leg']:.2f} | {100 * v['one_leg'] / STALL:.0f} % |")
    lines += ["", "| Arm pose | Joint | Torque [N m] | % of stall |", "|---|---|---|---|"]
    for pose, d in ch["arm_hold"].items():
        for n, v in d.items():
            lines.append(f"| {pose} | {n} | {v:.2f} | {100 * v / arm_stall:.0f} % |")
    res["worst_single_support_nm"] = worst
    text = "\n".join(lines)
    print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = "" if a.config == "chhaya.yaml" else "_" + os.path.splitext(a.config)[0].replace("chhaya_", "")
        open(os.path.join(a.out, f"servo_load{tag}.json"), "w").write(json.dumps(res, indent=2))
        open(os.path.join(a.out, f"SERVO_LOAD{tag.upper()}.md"), "w").write("# Servo load check\n\n" + text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
