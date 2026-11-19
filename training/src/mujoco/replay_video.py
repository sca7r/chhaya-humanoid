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
MESHES = os.path.join(os.path.dirname(__file__), "..", "..", "robot_hardware", "robot_description", "meshes")
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


def _stl_parts(path):
    """Triangles of a binary STL grouped into connected solids, largest first."""
    import struct
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    v = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("c", "<u2")]), count=n)["v"]
    _, idx = np.unique(np.round(v.reshape(-1, 3), 4), axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    par = np.arange(idx.max() + 1)

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for t in idx:
        r = [root(x) for x in t]
        par[r[1]] = par[r[2]] = r[0]
    lab = np.array([root(t[0]) for t in idx])
    return [v[lab == k] for k in sorted(set(lab), key=lambda k: -(lab == k).sum())]


def _styled_xml(src):
    """_dressed_xml (sky, checker floor) plus the colour scheme; the chest mesh is split into frame, head, rack."""
    import re
    import struct
    import tempfile
    path = _dressed_xml(src)
    x = open(path).read()
    d = os.path.dirname(path)
    # chest_link.stl holds three solids: the chest frame (largest box), the halo head and the electronics rack
    parts = _stl_parts(os.path.join(MESHES, "chest_link.stl"))
    names = []
    for i, tris in enumerate(parts):
        f = os.path.join(d, f"chest_part{i}.stl")
        with open(f, "wb") as fh:
            fh.write(b"\0" * 80 + struct.pack("<I", len(tris)))
            for t in tris:
                fh.write(struct.pack("<12fH", 0, 0, 0, *t.reshape(-1), 0))
        names.append(f)
    zc = [float(t.reshape(-1, 3)[:, 2].mean()) for t in parts]
    head, rack = int(np.argmax(zc)), int(np.argmin(zc))           # head on top, rack under the chest
    mat = {head: "v_print", rack: "v_print"}
    meshes = "".join(f'<mesh name="v_chest{i}" file="{f}" scale="0.001 0.001 0.001"/>' for i, f in enumerate(names))
    geoms = "".join(f'<geom type="mesh" mesh="v_chest{i}" contype="0" conaffinity="0" group="1" material="{mat.get(i, "v_print")}"/>'
                    for i in range(len(names)))
    # the chest_link frame is the CAD frame (no joint offset), so the stand-ins go in as they are (mm)
    for name, m in (("battery", "v_batt"), ("computer", "v_pcb")):
        meshes += f'<mesh name="v_{name}" file="{os.path.abspath(os.path.join(EXTRAS, name + ".stl"))}" scale="0.001 0.001 0.001"/>'
        geoms += f'<geom type="mesh" mesh="v_{name}" contype="0" conaffinity="0" group="1" mass="0" material="{m}"/>'
    x = x.replace("<asset>", "<asset>" + meshes + _MATERIALS, 1)
    x, n = re.subn(r'<geom type="mesh" mesh="chest_link" [^>]*/>', geoms, x, count=1)
    assert n == 1, "chest_link geom not found"
    x = re.sub(r'(<geom [^>]*mesh="[^"]*_servos"[^>]*?)rgba="[^"]*"', r'\1material="v_servo"', x)
    x = re.sub(r'(<geom [^>]*mesh="[lr]_(shoulder|upper_arm|elbow|forearm)_link"[^>]*?)rgba="[^"]*"', r'\1material="v_print"', x)

    def leg(m):
        g = m.group(0)
        return g.replace("/>", f' material="{"v_joint" if any(k in m.group(1) for k in _LEG_JOINTS) else "v_print"}"/>')
    x = re.sub(r'<geom [^>]*mesh="((?:[lr]_\w+_link_1)|base_link)"[^>]*/>', leg, x)
    x = x.replace("<worldbody>", "<worldbody>" + _LIGHT, 1)
    out = os.path.join(tempfile.mkdtemp(prefix="chhaya_look_"), "chhaya.xml")
    open(out, "w").write(x)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", required=True)
    p.add_argument("--gif", default=None, help="also a small GIF (GitHub shows GIFs inline)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--still", default=None, help="write one PNG of frame 0 here and stop (look check)")
    p.add_argument("--spacing", type=float, default=0.8, help="crowd: grid spacing of the spawn points [m]")
    a = p.parse_args()
    rec = np.load(a.npz)
    traj, names, dt = rec["traj"], [str(n) for n in rec["joint_names"]], float(rec["dt"])
    if traj.ndim == 2:
        traj = traj[:, None]                       # (T, robots, D)
    crowd = traj.shape[1] > 1
    if crowd:                                      # positions are relative to each robot's spawn: lay spawns on a grid
        k = traj.shape[1]; cols = math.ceil(math.sqrt(k))
        grid = np.array([[(i % cols) * a.spacing, (i // cols) * a.spacing, 0.0] for i in range(k)])
        traj = traj.copy(); traj[:, :, :3] += grid - grid.mean(0)
    m = mujoco.MjModel.from_xml_path(_styled_xml(XML))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(a.width, 640), max(a.height, 480)
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    J = len(names)
    r = mujoco.Renderer(m, a.height, a.width, max_geom=20000)
    cam = mujoco.MjvCamera()
    w, x, y, z = traj[0, 0, 3:7]
    if crowd:                                      # static-ish high camera over the whole group
        extent = np.ptp(traj[0, :, :2], axis=0).max() + 1.0
        cam.distance, cam.elevation, cam.azimuth = 1.1 * extent, -30.0, 135.0
    else:                                          # side view following the robot
        cam.distance, cam.elevation = 0.95, -12.0
        cam.azimuth = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) + 140.0   # three-quarter front
    look = traj[0, :, :3].mean(0)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    frames = []
    for row in traj:
        for i, ri in enumerate(row):
            d.qpos[:7] = ri[:7]                    # free joint: position, quaternion (w, x, y, z)
            d.qpos[adr] = ri[7:7 + J]
            mujoco.mj_kinematics(m, d)
            if i == 0:
                look += (0.02 if crowd else 0.08) * (row[:, :3].mean(0) - look)   # smoothed follow
                cam.lookat[:] = (look[0], look[1], 0.2)
                r.update_scene(d, cam)
            else:                                  # the other robots: their geoms only (floor and lights once)
                mujoco.mjv_addGeoms(m, d, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        frames.append(r.render())
        if a.still:
            imageio.imwrite(a.still, frames[0])
            return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=round(1 / dt), quality=8, macro_block_size=8)
    if a.gif:
        small = [f[::4, ::4] for f in frames[::2]]  # a quarter of the size, 25 fps (GitHub inline)
        imageio.mimwrite(a.gif, small, duration=2 * dt * 1000, loop=0)
    print(f"[replay] {len(frames)} frames -> {a.out}" + (f", {a.gif}" if a.gif else ""))


if __name__ == "__main__":
    main()
