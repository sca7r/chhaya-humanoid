"""replay_video.py -- render a motion recorded in Isaac Lab (train_supermimic.py --eval --record) in MuJoCo.

Kinematic replay: MuJoCo only draws the poses the training simulator produced (no physics here), so the
video shows exactly what the policy did in Isaac Lab / Newton. Side camera following the robot.

    MUJOCO_GL=osmesa python -m src.mujoco.replay_video runs/x/walk.npz --out videos/walk.mp4 --gif videos/walk.gif
"""
import argparse
import math
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

from src.mujoco.record_reel import _dressed_xml

XML = os.path.join(os.path.dirname(__file__), "..", "..", "envs", "assets", "robotV2", "chhaya.xml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", required=True)
    p.add_argument("--gif", default=None, help="also a small GIF (GitHub shows GIFs inline)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--title", default="")
    a = p.parse_args()
    rec = np.load(a.npz)
    traj, names, dt = rec["traj"], [str(n) for n in rec["joint_names"]], float(rec["dt"])
    m = mujoco.MjModel.from_xml_path(_dressed_xml(XML))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(a.width, 640), max(a.height, 480)
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    J = len(names)
    r = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 1.3, -15.0
    w, x, y, z = traj[0, 3:7]
    cam.azimuth = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) + 90.0
    look = traj[0, :3].copy()
    frames = []
    for row in traj:
        d.qpos[:7] = row[:7]                       # free joint: position, quaternion (w, x, y, z)
        d.qpos[adr] = row[7:7 + J]
        mujoco.mj_kinematics(m, d)
        look += 0.08 * (row[:3] - look)            # smoothed follow
        cam.lookat[:] = (look[0], look[1], 0.2)
        r.update_scene(d, cam)
        frames.append(r.render())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=round(1 / dt), quality=8, macro_block_size=8)
    if a.gif:
        small = [f[::4, ::4] for f in frames[::2]]  # a quarter of the size, 25 fps (GitHub inline)
        imageio.mimwrite(a.gif, small, duration=2 * dt * 1000, loop=0)
    print(f"[replay] {len(frames)} frames -> {a.out}" + (f", {a.gif}" if a.gif else ""))


if __name__ == "__main__":
    main()
