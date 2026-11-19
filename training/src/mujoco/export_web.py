"""export_web.py -- write the browser demo's assets (web/assets/) from an exported policy bundle.

    python -m src.mujoco.export_web --bundle exported/chhaya_final

Writes policy.onnx, chhaya.xml (the styled render model, meshes from web/meshes and web/extras) and
robot.json: every number web/sim.mjs needs, read from BipedEnvV2 itself so the browser cannot drift from the
MuJoCo check. trace.json (--trace) is a closed-loop reference run for web/test_sim.mjs.
"""
import argparse
import dataclasses
import json
import math
import os
import re

import numpy as np

from envs.mujoco.biped_env_v2 import (ACTION_FILTER_ALPHA, DECIMATION, IMU_OBS_DELAY, JOINT_NAMES, PHYSICS_DT,
                                      SERVO_OBS_DELAY, BipedEnvV2)
from src.mujoco.policy_gate import Policy, _run_config_path
from src.mujoco.replay_video import XML, _styled_xml
from utils import Config

WEB = os.path.join(os.path.dirname(__file__), "..", "..", "..", "web")


def make_env(bundle, policy):
    ec = Config(_run_config_path(bundle)).train.env_config
    ec.backlash = {**(ec.backlash or {}), "enabled": False}      # as the MuJoCo check runs it
    assert SERVO_OBS_DELAY == 0, "web/sim.mjs implements the IMU delay only"
    return BipedEnvV2(history_size=policy.hist, use_history=True, critic_has_privileged_info=True, env_config=ec)


def lst(x):
    return None if x is None else [float(v) for v in np.asarray(x).ravel()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", default=os.path.join(WEB, "assets"))
    p.add_argument("--trace", type=int, default=0, help="also write a closed-loop reference run of N steps")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    policy = Policy(a.bundle)
    env = make_env(a.bundle, policy)
    import onnx      # one file for the browser: the bundle keeps the weights in policy.onnx.data
    onnx.save(onnx.load(os.path.join(a.bundle, "policy.onnx")), os.path.join(a.out, "policy.onnx"))

    # render + physics model: the styled XML, mesh paths relative to web/
    xml = open(_styled_xml(XML)).read()
    xml = re.sub(r'file="[^"]*/robot_description/meshes/([^"/]+)"', r'file="meshes/\1"', xml)
    xml = re.sub(r'file="[^"]*/checks/extras/([^"/]+)"', r'file="extras/\1"', xml)
    xml = re.sub(r'\s+meshdir="[^"]*"', "", xml)
    assert not re.search(r'(file|meshdir)="/', xml), "absolute path left in the web XML"
    open(os.path.join(a.out, "chhaya.xml"), "w").write(xml)

    k = env.servo.kernel
    ranges, shape = env.command_limit()
    g = env.gait_clock
    robot = {
        "joints": JOINT_NAMES, "arms": list(env._arm_names),
        "home": lst(env._residual_home), "scale": lst(env._residual_scale), "scale_neg": lst(env._residual_scale_neg),
        "jlim_min": lst(env._jlim_min), "jlim_max": lst(env._jlim_max), "spawn_dz": float(env._spawn_dz),
        "arm_rest": lst(env._arm_rest), "arm_scale": lst(env._arm_scale),
        "arm_lim_min": lst(env._arm_lim_min), "arm_lim_max": lst(env._arm_lim_max),
        "kp": float(k.kp.flatten()[0]), "vin": float(k.vin.flatten()[0]), "deadband": float(k.deadband.flatten()[0]),
        "torque_scale": float(k.torque_scale.flatten()[0]),
        "bam": {f.name: getattr(k.p, f.name) for f in dataclasses.fields(k.p) if f.name not in ("source", "actuator")},
        "ground_friction": getattr(env.config, "mujoco_ground_friction", None),
        "physics_dt": PHYSICS_DT, "decimation": DECIMATION, "alpha": ACTION_FILTER_ALPHA, "imu_delay": IMU_OBS_DELAY,
        "history": policy.hist, "obs_mean": lst(policy.mean), "obs_std": lst(policy.std),
        "gait": {"vx": lst(g.axes[0]), "vy": lst(g.axes[1]), "wz": lst(g.axes[2]), "freq": lst(g.freq),
                 "stand": float(env.gait_stand_threshold)},
        "ranges": {key: [float(v) for v in ranges[key]] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "shape": shape, "tilt_limit": 0.784,
    }
    json.dump(robot, open(os.path.join(a.out, "robot.json"), "w"))

    if a.trace:          # reference run: phase 0, fixed command, the policy in the loop
        env.reset(seed=0)                       # web/sim.js reset(): the same steps, in the same order
        env.gait_phase = 0.0
        env.set_command(0.15, 0.0, 0.0)
        env._history.extend([np.zeros(env._base_actor_dim, np.float32)] * policy.hist)
        obs = env._build_obs()
        rows = []
        for _ in range(a.trace):
            act = policy(obs["actor"])
            rows.append({"obs": lst(obs["actor"][-env._base_actor_dim:]), "act": lst(act),
                         "qpos": lst(env.data.qpos)})
            obs, _, term, _, _ = env.step(act)
            if term:
                break
        json.dump({"cmd": [0.15, 0.0, 0.0], "rows": rows}, open(os.path.join(a.out, "trace.json"), "w"))
    print(f"[export_web] {a.out}: policy.onnx, chhaya.xml, robot.json" + (", trace.json" if a.trace else ""))


if __name__ == "__main__":
    main()
