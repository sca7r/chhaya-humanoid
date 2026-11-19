"""replay_video.py -- render a motion recorded in Isaac Lab (train_supermimic.py --eval --record) in MuJoCo.

Kinematic replay: MuJoCo only draws the poses the training simulator produced (no physics here), so the
video shows exactly what the policy did in Isaac Lab / Newton. Several recordings play back to back (--labels).

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
# battery and Raspberry Pi 5 stand-ins from the CAD (their mass is in chest_link already; the meshes are not)
EXTRAS = os.path.join(os.path.dirname(__file__), "..", "..", "robot_hardware", "cad", "parametric", "checks", "extras")

# Render-only look (the generated chhaya.xml is not touched): printed parts white, servos black, electronics blue.
_MATERIALS = """
    <material name="v_print" rgba="0.92 0.92 0.90 1" specular="0.25" shininess="0.35"/>
    <material name="v_joint" rgba="0.20 0.21 0.23 1" specular="0.4" shininess="0.5"/>
    <material name="v_servo" rgba="0.06 0.06 0.07 1" specular="0.5" shininess="0.7"/>
    <material name="v_pcb" rgba="0.10 0.35 0.85 1" specular="0.4" shininess="0.6"/>
    <material name="v_batt" rgba="0.16 0.17 0.19 1" specular="0.6" shininess="0.8"/>"""
_LIGHT = '<light name="v_key" directional="true" castshadow="true" pos="1 -1 3" dir="-0.4 0.4 -1" diffuse="0.55 0.55 0.55"/>'
# leg links are one solid each (servo + bracket merged): joint blocks dark, limb segments white
_LEG_JOINTS = ("hip_upper", "hip_lower", "ankle")


def _styled_xml(src):
    """_dressed_xml (sky, checker floor) plus the colour scheme and the battery / Pi stand-ins on the chest."""
    import re
    path = _dressed_xml(src)
    x = open(path).read()
    meshes, extra = "", ""
    for name, m in (("battery", "v_batt"), ("computer", "v_pcb")):    # chest_link frame = CAD frame (mm)
        meshes += f'<mesh name="v_{name}" file="{os.path.abspath(os.path.join(EXTRAS, name + ".stl"))}" scale="0.001 0.001 0.001"/>'
        extra += f'<geom type="mesh" mesh="v_{name}" contype="0" conaffinity="0" group="1" mass="0" material="{m}"/>'
    x = x.replace("<asset>", "<asset>" + meshes + _MATERIALS, 1)
    x, n = re.subn(r'(<geom type="mesh" mesh="chest_link" [^>]*/>)', rf"\1{extra}", x, count=1)
    assert n == 1, "chest_link geom not found"
    x = re.sub(r'(<geom [^>]*mesh="[^"]*_servos"[^>]*?)rgba="[^"]*"', r'\1material="v_servo"', x)
    x = re.sub(r'(<geom [^>]*mesh="(chest_link|[lr]_(shoulder|upper_arm|elbow|forearm)_link)"[^>]*?)rgba="[^"]*"',
               r'\1material="v_print"', x)

    def leg(m):
        return m.group(0).replace("/>", f' material="{"v_joint" if any(k in m.group(1) for k in _LEG_JOINTS) else "v_print"}"/>')
    x = re.sub(r'<geom [^>]*mesh="((?:[lr]_\w+_link_1)|base_link)"[^>]*/>', leg, x)
    open(path, "w").write(x.replace("<worldbody>", "<worldbody>" + _LIGHT, 1))
    return path


def _pose(m, d, adr, J, ri):
    d.qpos[:7] = ri[:7]                            # free joint: position, quaternion (w, x, y, z)
    d.qpos[adr] = ri[7:7 + J]
    mujoco.mj_kinematics(m, d)


def _caption(frame, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    ImageDraw.Draw(im).text((24, 18), text, fill="white", stroke_width=3, stroke_fill="black",
                              font_size=round(frame.shape[0] / 20))
    return np.asarray(im)


def _segment(m, d, r, a, npz, label):
    """Frames of one recording: crowd (high camera over the grid) or one robot (followed; --iso: fixed isometric)."""
    rec = np.load(npz)
    traj, names, dt = rec["traj"], [str(n) for n in rec["joint_names"]], float(rec["dt"])
    if traj.ndim == 2:
        traj = traj[:, None]                       # (T, robots, D)
    if a.seconds:
        traj = traj[: round(a.seconds / dt)]
    crowd = traj.shape[1] > 1
    if crowd:                                      # positions are relative to each robot's spawn: lay spawns on a grid
        k = traj.shape[1]; cols = math.ceil(math.sqrt(k))
        grid = np.array([[(i % cols) * a.spacing, (i // cols) * a.spacing, 0.0] for i in range(k)])
        traj = traj.copy(); traj[:, :, :3] += grid - grid.mean(0)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    J = len(names)
    cam = mujoco.MjvCamera()
    w, x, y, z = traj[0, 0, 3:7]
    if crowd:                                      # static-ish high camera over the whole group
        extent = np.ptp(traj[0, :, :2], axis=0).max() + 1.0
        cam.distance, cam.elevation, cam.azimuth = 1.1 * extent, -30.0, 135.0
    elif a.iso:                                    # fixed world direction: the robot's own heading and turns show
        cam.distance, cam.elevation, cam.azimuth = 1.05, -35.26, 135.0
    else:                                          # side view following the robot
        cam.distance, cam.elevation = 0.95, -12.0
        cam.azimuth = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) + 140.0   # three-quarter front
    look = traj[0, :, :3].mean(0)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    for row in traj:
        for i, ri in enumerate(row):
            _pose(m, d, adr, J, ri)
            if i == 0:
                look += (0.02 if crowd else 0.08) * (row[:, :3].mean(0) - look)   # smoothed follow
                cam.lookat[:] = (look[0], look[1], 0.2)
                r.update_scene(d, cam)
            else:                                  # the other robots: their geoms only (floor and lights once)
                mujoco.mjv_addGeoms(m, d, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        f = r.render()
        yield _caption(f, label) if label else f


def _still(m, d, r, a):
    """One frame of the first recording from four sides (front, side, back, isometric), tiled 2 x 2."""
    rec = np.load(a.npz[0])
    traj, names = rec["traj"], [str(n) for n in rec["joint_names"]]
    ri = (traj[:, 0] if traj.ndim == 3 else traj)[round(a.still / float(rec["dt"]))]
    _pose(m, d, [m.jnt_qposadr[m.joint(n).id] for n in names], len(names), ri)
    w, x, y, z = ri[3:7]
    head = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (ri[0], ri[1], 0.24)
    tiles = []
    for az, el, label in ((180, -8, "front"), (90, -8, "side"), (0, -8, "back"), (135, -35.26, "isometric")):
        cam.distance, cam.elevation, cam.azimuth = 1.0, el, head + az
        r.update_scene(d, cam)
        tiles.append(_caption(r.render(), label))
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+", help="one or more recordings, played one after another")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="*", default=None, help="a caption per recording")
    p.add_argument("--seconds", type=float, default=None, help="keep the first N s of each recording")
    p.add_argument("--iso", action="store_true", help="single robot: fixed isometric camera")
    p.add_argument("--still", type=float, default=None, help="write one PNG: the robot at T s from four sides")
    p.add_argument("--gif", default=None, help="also a small GIF (GitHub shows GIFs inline)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--spacing", type=float, default=0.8, help="crowd: grid spacing of the spawn points [m]")
    a = p.parse_args()
    m = mujoco.MjModel.from_xml_path(_styled_xml(XML))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(a.width, 640), max(a.height, 480)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, a.height, a.width, max_geom=20000)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.still is not None:
        imageio.imwrite(a.out, _still(m, d, r, a))
        print(f"[replay] still -> {a.out}")
        return
    dt = float(np.load(a.npz[0])["dt"])
    labels = a.labels or [None] * len(a.npz)
    frames = [f for npz, label in zip(a.npz, labels) for f in _segment(m, d, r, a, npz, label)]
    imageio.mimwrite(a.out, frames, fps=round(1 / dt), quality=8, macro_block_size=8)
    if a.gif:
        small = [f[::4, ::4] for f in frames[::2]]  # a quarter of the size, 25 fps (GitHub inline)
        imageio.mimwrite(a.gif, small, duration=2 * dt * 1000, loop=0)
    print(f"[replay] {len(frames)} frames -> {a.out}" + (f", {a.gif}" if a.gif else ""))


if __name__ == "__main__":
    main()
