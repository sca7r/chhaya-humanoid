"""Render the robot in a few poses (headless: MUJOCO_GL=egl).

    MUJOCO_GL=egl python -m chhaya.render --variant sts3215 --out renders/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from chhaya.model import model_path

POSES = {
    "zero": {},
    "t_pose": {"l_shoulder_roll": 1.57, "r_shoulder_roll": 1.57},
    "wave": {"r_shoulder_roll": 0.35, "r_shoulder_pitch": -2.6, "r_elbow": -1.2, "r_arm_yaw": 0.4,
             "l_shoulder_pitch": 0.2},
    "reach": {"l_shoulder_pitch": -1.4, "r_shoulder_pitch": -1.4, "l_elbow": -0.4, "r_elbow": -0.4},
}


def render(variant: str, out: Path, size=(720, 540)):
    m = mujoco.MjModel.from_xml_path(str(model_path(variant)))
    m.vis.global_.offwidth, m.vis.global_.offheight = size
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size[1], size[0])
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for pname, pose in POSES.items():
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.30
        for j, v in pose.items():
            d.qpos[m.joint(j).qposadr[0]] = v
        mujoco.mj_forward(m, d)
        for vname, (az, el) in {"front": (180, -5), "side": (90, -5), "persp": (145, -18)}.items():
            cam.lookat[:] = [0.0, 0.0, 0.25]
            cam.distance, cam.azimuth, cam.elevation = 0.95, az, el
            r.update_scene(d, cam, opt)
            img = r.render()
            f = out / f"{variant}_{pname}_{vname}.png"
            from PIL import Image
            Image.fromarray(img).save(f)
            files.append(f)
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sts3215", choices=["sts3215"])
    ap.add_argument("--out", type=Path, default=Path("renders"))
    a = ap.parse_args()
    for f in render(a.variant, a.out):
        print(f)
