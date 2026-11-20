"""
Pictures of the whole robot from the generated MuJoCo model (visor and jaws coloured separately).

    MUJOCO_GL=egl python tools/render_robot.py [--out media/chhaya]
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "parametric"))
sys.path.insert(0, str(ROOT / "tools"))
import chhaya_cad as ub                                  # noqa: E402
from build_description import frame_origin_mm            # noqa: E402

CAD = ub.CHECKS
WHITE, DARK, BLACK, LEG = "0.95 0.95 0.96 1", "0.20 0.21 0.24 1", "0.05 0.05 0.07 1", "0.30 0.31 0.34 1"


def scene(hw, train, tmp):
    xml = Path(train) / "envs/assets/robotV2/chhaya.xml"
    tree = ET.parse(xml)
    root = tree.getroot()
    asset = root.find("asset")
    for me in asset.findall("mesh"):
        f = me.get("file")
        name = Path(f).name
        me.set("file", str(Path(hw) / "robot_description/meshes" / name))
    def add(link, parts, rgba, tag):
        o = frame_origin_mm(link)
        m = trimesh.util.concatenate([trimesh.load(ub.STL_DIR / f"{p}.stl") for p in parts])
        m.apply_translation(-o)
        f = tmp / f"{link}_{tag}.stl"
        m.export(f)
        ET.SubElement(asset, "mesh", name=f"{link}_{tag}", file=str(f), scale="0.001 0.001 0.001")
        b = root.find(f".//body[@name='{link}']")
        ET.SubElement(b, "geom", type="mesh", mesh=f"{link}_{tag}", contype="0", conaffinity="0", group="1", rgba=rgba)
    # replace the single-colour link meshes of the chest and forearms by coloured groups
    for link, groups in [("chest_link", [(["chest_frame", "head"], WHITE, "a"), (["head_visor"], BLACK, "b")])] + \
            [(f"{s}_forearm_link", [([f"{s}_forearm", f"{s}_claw_base"], WHITE, "a"), ([f"{s}_claw_jaw"], DARK, "b")]) for s in "lr"]:
        b = root.find(f".//body[@name='{link}']")
        for g in list(b.findall("geom")):
            if g.get("mesh") == link:
                b.remove(g)
        for parts, rgba, tag in groups:
            add(link, parts, rgba, tag)
    for g in root.iter("geom"):
        if g.get("type") == "mesh" and g.get("rgba") is None:
            g.set("rgba", LEG)
        elif g.get("rgba", "").startswith("0.93"):
            g.set("rgba", WHITE)
        if g.get("type") == "plane":
            g.set("rgba", "0.93 0.93 0.94 1")
            g.attrib.pop("material", None)
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    for t in list(vis):
        vis.remove(t)
    ET.SubElement(vis, "global", offwidth="1600", offheight="1800")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.5 0.5 0.5", specular="0.1 0.1 0.1")
    for t in asset.findall("texture"):
        if t.get("type") == "skybox":
            asset.remove(t)
    ET.SubElement(asset, "texture", type="skybox", builtin="flat", rgb1="0.96 0.96 0.97", rgb2="0.96 0.96 0.97", width="8", height="8")
    p = tmp / "scene.xml"
    tree.write(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default=str(ROOT))
    ap.add_argument("--train", default=str(ROOT.parent / "training"))
    ap.add_argument("--out", default=str(ROOT / "media" / "chhaya"))
    a = ap.parse_args()
    out = Path(a.out)
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(scene(a.hw, a.train, tmp)))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.30
    for j, v in {"l_shoulder_roll": 0.18, "r_shoulder_roll": 0.40, "l_elbow": -0.35, "r_elbow": -0.75, "r_arm_yaw": -0.3}.items():
        d.qpos[m.joint(j).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[:3] = 1
    cam = mujoco.MjvCamera()
    views = {"front": (1000, 1400, 155, -8, 0.86, [0, 0, 0.275]), "back": (1000, 1400, 25, -10, 0.86, [0, 0, 0.275]),
             "head": (1000, 700, 158, -4, 0.27, [0.01, 0.0, 0.47]), "head_back": (1000, 700, 12, -6, 0.25, [0.01, 0.0, 0.475])}
    ims = {}
    for name, (W, H, az, el, dist, la) in views.items():
        r = mujoco.Renderer(m, H, W)
        cam.lookat[:] = la
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(d, cam, opt)
        ims[name] = Image.fromarray(r.render())
        r.close()
    c = Image.new("RGB", (3000, 1400), (245, 245, 247))
    c.paste(ims["front"], (0, 0)); c.paste(ims["back"], (1000, 0))
    c.paste(ims["head"], (2000, 0)); c.paste(ims["head_back"], (2000, 700))
    c.save(out / "overview.png")
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print("wrote", out / "overview.png")


if __name__ == "__main__":
    main()
