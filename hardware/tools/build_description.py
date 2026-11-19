"""
Build Chhaya's robot description from the parametric CAD (cad/parametric/chhaya_cad.py).

The leg links come from the Fusion export (robot_description/urdf/Robot.urdf and the training
repo's robot_v2.xml, both kept as the legs-only rig); this adds the chest, head and arm links and
writes the whole robot:
  hardware/robot_description/urdf/chhaya.urdf          whole robot
  hardware/robot_description/meshes/<link>*.stl         visual, servo and collision meshes (mm, link frames)
  hardware/robot_description/chhaya_mass.json           mass budget of the generated links
  training/envs/assets/robotV2/chhaya.xml               MuJoCo model, whole robot

Mass properties come from the CAD: printed parts at an effective PLA density (walls + infill),
each STS3215 as a uniform box of its real mass, the claw micro servos likewise. Usage:

  python tools/build_description.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "cad" / "parametric"))
import chhaya_cad as ub  # noqa: E402

PLA_EFFECTIVE_DENSITY = 1.24e3 * 0.75   # kg/m^3: PLA 1.24 g/cm^3, ~75% fill for thin-walled parts
HEAD_PAYLOAD_KG = 0.0                   # whatever you mount on the head ring (e.g. 0.004 for a Pi camera)
CABLE_KG_PER_LINK = 0.004               # servo lead share per moving link
# Torso electronics and wiring not in any CAD part (2026-11-19, estimates at the upper end; weigh to replace):
# power PCB + servo bus adapter 35 g, IMU board 4 g, main power wiring + XT60 + switch 30 g, screws and heat-set
# inserts outside the servos 25 g, the 12 leg servo leads (they bundle into the torso) 48 g. In the torso cavity
# (82 x 32 x 28 mm), centred on the torso's centre of mass. Training adds -50..+150 g on top (payload DR).
TORSO_ELECTRONICS_KG = 0.035 + 0.004 + 0.030 + 0.025 + 0.048
TORSO_ELECTRONICS_COM_MM = (2.4, -1.7, 38.2)
TORSO_ELECTRONICS_SIZE_MM = (82.0, 32.0, 28.0)

# The original torso inertial (Robot.urdf base_link, kept as is) was computed by Fusion over the torso model,
# which includes stand-in blocks for the original electronics inside the rim tray (above z = 70.2 mm). Chhaya
# puts the electronics rack, battery and computer there instead, modelled explicitly, so the stand-ins' share
# is removed (negative item on chest_link). Measured on base_link.stl: 68.9 cm3 above the shelf, centroid
# (10.5, 0.1, 81.3) mm, at the base_link's average density (0.633 kg / 449.6 cm3). Weigh the real torso to
# replace this estimate.
TORSO_STANDIN_KG = 0.633299 * 68.9 / 449.6
TORSO_STANDIN_COM_MM = (10.5, 0.1, 81.3)
TORSO_STANDIN_SIZE_MM = (40.0, 59.0, 24.0)

# Joint limits (rad). Sign conventions: shoulder_roll/arm_yaw axes are mirrored on the right,
# so for both arms: +roll = abduct (arm out), +pitch = arm backward, -pitch = arm forward/up,
# -elbow = flex forward, +yaw = rotate the upper arm outward.
LIMITS = {
    "shoulder_pitch": (-3.0, 1.0),
    "shoulder_roll": (0.05, 2.6),
    "arm_yaw": (-1.57, 1.57),
    "elbow": (-2.0, 0.1),
}

# link name -> (parent link, joint name or None, printed parts, servos carried)
def links_for():
    L = {
        "chest_link": ("base_link", None, ["chest_frame", "electronics_rack", "head", "head_visor"],
                       ["l_shoulder_pitch", "r_shoulder_pitch"]),
    }
    for s in "lr":
        L[f"{s}_shoulder_link"] = ("chest_link", f"{s}_shoulder_pitch", [f"{s}_shoulder_bracket", f"{s}_roll_idler_cap"], [f"{s}_shoulder_roll"])
        L[f"{s}_upper_arm_link"] = (f"{s}_shoulder_link", f"{s}_shoulder_roll", [f"{s}_upper_arm"], [f"{s}_arm_yaw"])
        L[f"{s}_elbow_link"] = (f"{s}_upper_arm_link", f"{s}_arm_yaw", [f"{s}_elbow_link", f"{s}_elbow_idler_cap"], [f"{s}_elbow"])
        L[f"{s}_forearm_link"] = (f"{s}_elbow_link", f"{s}_elbow", [f"{s}_forearm", f"{s}_claw_base", f"{s}_claw_jaw"], [])
    return L


DARK_PARTS = ("head_visor",)

UPPER_JOINT_ORDER = [f"{s}_{j}" for j in ("shoulder_pitch", "shoulder_roll", "arm_yaw", "elbow") for s in "lr"]


def extras_for(variant, link):
    """Bought parts that are not STS3215s (the claw micro servos): [(name, mass kg, lo mm, hi mm)].
    The claw is not a simulated joint: its servo and jaw are fixed mass on the forearm."""
    info = json.loads((ub.CHECKS / "parts.json").read_text())
    return [(n, e["mass_kg"], np.array(e["bounds_mm"][0]), np.array(e["bounds_mm"][1]))
            for n, e in info.get("extras", {}).items() if e["link"] == link]


def frame_origin_mm(link):
    _, joint, _, _ = links_for()[link]
    return np.zeros(3) if joint is None else np.array(ub.JOINTS[joint][0], float)


def box_inertia(m, size):
    a, b, c = size
    return np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])


def combine(items):
    """items: list of (mass kg, com m (3,), inertia about own com (3,3)) -> total."""
    M = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / M
    I = np.zeros((3, 3))
    for m, c, Ic in items:
        d = c - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, com, I


def cad_mass(step_path, density=PLA_EFFECTIVE_DENSITY):
    """Exact mass properties from the B-rep (mm in, SI out)."""
    import cadquery as cq
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    shp = cq.importers.importStep(str(step_path)).val().wrapped
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shp, g)
    vol = g.Mass() * 1e-9
    c = g.CentreOfMass()
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]) * 1e-15 * density
    return vol * density, np.array([c.X(), c.Y(), c.Z()]) / 1000, I


def mass_props(variant, cad_dir):
    info = json.loads((cad_dir / "parts.json").read_text())
    sv = ub.SERVOS[variant]
    out, budget = {}, {}
    for link, (_, _, parts, servos) in links_for().items():
        o = frame_origin_mm(link) / 1000.0
        items = []
        for p in parts:
            m, c, I = cad_mass(ub.STEP_DIR / f"{p}.step")
            items.append((m, c - o, I))
            budget[p] = round(m * 1000, 1)
        for s in servos:
            lo, hi = (np.array(b) / 1000 for b in info["servos"][s]["case_bounds_mm"])
            items.append((sv.mass_kg, (lo + hi) / 2 - o, box_inertia(sv.mass_kg, hi - lo)))
            budget[f"servo_{s}"] = sv.mass_kg * 1000
        for n, m, lo, hi in extras_for(variant, link):
            items.append((m, (lo + hi) / 2000 - o, box_inertia(m, (hi - lo) / 1000)))
            budget[n] = m * 1000
        if link == "chest_link":
            c = np.array(TORSO_STANDIN_COM_MM) / 1000
            items.append((-TORSO_STANDIN_KG, c - o, box_inertia(-TORSO_STANDIN_KG, np.array(TORSO_STANDIN_SIZE_MM) / 1000)))
            budget["original_torso_standins_removed"] = round(-TORSO_STANDIN_KG * 1000, 1)
            c = np.array(TORSO_ELECTRONICS_COM_MM) / 1000
            items.append((TORSO_ELECTRONICS_KG, c - o, box_inertia(TORSO_ELECTRONICS_KG, np.array(TORSO_ELECTRONICS_SIZE_MM) / 1000)))
            budget["torso_electronics_wiring"] = round(TORSO_ELECTRONICS_KG * 1000, 1)
            # head payload (camera etc.) at the ring centre; 0 by default -- set HEAD_PAYLOAD_KG
            if HEAD_PAYLOAD_KG > 0:
                cam = np.array(ub.head_center()) / 1000
                items.append((HEAD_PAYLOAD_KG, cam - o, np.eye(3) * 1e-8))
                budget["head_payload"] = HEAD_PAYLOAD_KG * 1000
        else:
            items.append((CABLE_KG_PER_LINK, items[0][1], np.eye(3) * 1e-8))
            budget[f"cables_{link}"] = CABLE_KG_PER_LINK * 1000
        out[link] = combine(items)
    return out, budget


def split_for_collision(part, mesh):
    """MuJoCo collides each mesh as its convex hull. Concave parts are cut into pieces first so
    the hulls stay close to the real shape (the forearm's hull would otherwise fill the space
    between the fork and the claw and hit the hip when the leg abducts)."""
    cuts = {}
    zb = ub.ELBOW[2] - ub.FORK_D - ub.FORK_TB                    # forearm: fork | forearm+claw
    for s_ in "lr":
        cuts[f"{s_}_forearm"] = [zb]                              # fork | shell + wrist plate
    cuts["chest_frame"] = [ub.CH_Z0 - 3.0]                       # chest box | A-legs
    cuts["electronics_rack"] = [ub.DECK_Z0 - 0.5]                 # plate + posts | deck
    zs = cuts.get(part)
    if not zs:
        return [mesh]
    # vertex bands (1 mm overlap) -> one hull each; avoids needing a polygon library to cap slices
    v = mesh.vertices
    edges = [np.inf] + sorted(zs, reverse=True) + [-np.inf]
    pieces = []
    for hi, lo in zip(edges[:-1], edges[1:]):
        sel = v[(v[:, 2] <= hi + 1.0) & (v[:, 2] >= lo - 1.0)]
        if len(sel) >= 4:
            pieces.append(trimesh.Trimesh(vertices=sel, faces=np.zeros((0, 3), int), process=False))
    return pieces


def write_meshes(variant, cad_dir, mesh_dir):
    mesh_dir.mkdir(parents=True, exist_ok=True)
    """Per link: <link>.stl (printed parts), <link>_servos.stl (servo stand-ins, visual only) and
    one convex collision mesh per printed part (<link>_col<i>.stl): MuJoCo collides a mesh as its
    convex hull, so one hull per part keeps the chest from becoming one big blob."""
    names = {}
    for link, (_, _, parts, servos) in links_for().items():
        o = frame_origin_mm(link)
        ms = [trimesh.load(ub.STL_DIR / f"{p}.stl") for p in parts]
        dark = [mm for p, mm in zip(parts, ms) if p in DARK_PARTS]      # drawn with the servos (dark)
        m = trimesh.util.concatenate([mm for p, mm in zip(parts, ms) if p not in DARK_PARTS])
        m.apply_translation(-o)
        m.export(mesh_dir / f"{link}.stl")
        cols = []
        k = 0
        for p, mm in zip(parts, ms):
            for piece in split_for_collision(p, mm):
                c = piece.convex_hull
                c.apply_translation(-o)
                c.export(mesh_dir / f"{link}_col{k}.stl")
                cols.append(f"{link}_col{k}")
                k += 1
        sv = None
        ex = [n for n, _, _, _ in extras_for(variant, link)]
        if servos or ex or dark:
            sm = trimesh.util.concatenate(dark + [trimesh.load(cad_dir / "servo_standins" / f"{s}.stl") for s in servos]
                                          + [trimesh.load(cad_dir / "extras" / f"{n}.stl") for n in ex])
            sm.apply_translation(-o)
            sm.export(mesh_dir / f"{link}_servos.stl")
            sv = f"{link}_servos"
        names[link] = {"cols": cols, "servos": sv}
    return names


def fmt(v, n=6):
    return " ".join(f"{x:.{n}g}" for x in v)


def servo_boxes(variant, link):
    """Collision boxes for the servo cases a link carries: [(center m in link frame, half sizes m)]."""
    info = json.loads((ub.CHECKS / "parts.json").read_text())
    o = frame_origin_mm(link)
    out = []
    for s in links_for()[link][3]:
        lo, hi = (np.array(b) for b in info["servos"][s]["case_bounds_mm"])
        out.append((((lo + hi) / 2 - o) / 1000, (hi - lo) / 2000))
    for _, _, lo, hi in extras_for(variant, link):
        out.append((((lo + hi) / 2 - o) / 1000, (hi - lo) / 2000))
    return out


def urdf_block(variant, props, meshes):
    lines = [f"\n<!-- ===== Chest, head and arms: generated by hardware/tools/build_description.py ===== -->"]
    pkg = "package://Robot_description/meshes"
    for link, (parent, joint, _, _) in links_for().items():
        M, com, I = props[link]
        lines += [
            f'<link name="{link}">',
            '  <inertial>',
            f'    <origin xyz="{fmt(com)}" rpy="0 0 0"/>',
            f'    <mass value="{M:.6f}"/>',
            f'    <inertia ixx="{I[0,0]:.4e}" iyy="{I[1,1]:.4e}" izz="{I[2,2]:.4e}" ixy="{I[0,1]:.4e}" iyz="{I[1,2]:.4e}" ixz="{I[0,2]:.4e}"/>',
            '  </inertial>',
            f'  <visual><geometry><mesh filename="{pkg}/{link}.stl" scale="0.001 0.001 0.001"/></geometry><material name="silver"/></visual>',
        ]
        if meshes[link]["servos"]:
            lines.append(f'  <visual><geometry><mesh filename="{pkg}/{meshes[link]["servos"]}.stl" scale="0.001 0.001 0.001"/></geometry></visual>')
        for c in meshes[link]["cols"]:
            lines.append(f'  <collision><geometry><mesh filename="{pkg}/{c}.stl" scale="0.001 0.001 0.001"/></geometry></collision>')
        for ctr, half in servo_boxes(variant, link):
            lines.append(f'  <collision><origin xyz="{fmt(ctr)}" rpy="0 0 0"/><geometry><box size="{fmt(2 * half)}"/></geometry></collision>')
        lines.append('</link>')
        o = (frame_origin_mm(link) - (frame_origin_mm(parent) if parent != "base_link" else 0)) / 1000
        if joint is None:
            lines += [f'<joint name="{link.replace("_link", "")}_fixed" type="fixed">',
                      f'  <origin xyz="{fmt(o)}" rpy="0 0 0"/>',
                      f'  <parent link="{parent}"/>', f'  <child link="{link}"/>', '</joint>']
        else:
            lo, hi = LIMITS[joint[2:]]
            sv = ub.SERVOS[variant]
            lines += [f'<joint name="{joint}" type="revolute">',
                      f'  <origin xyz="{fmt(o)}" rpy="0 0 0"/>',
                      f'  <parent link="{parent}"/>', f'  <child link="{link}"/>',
                      f'  <axis xyz="{fmt(ub.JOINTS[joint][1])}"/>',
                      f'  <limit lower="{lo}" upper="{hi}" effort="{sv.stall_nm}" velocity="{4.9 if variant == "sts3215" else 8.0}"/>',
                      '</joint>']
        lines.append("")
    return "\n".join(lines)


def write_urdf(variant, props, meshes, hw_root, out_urdf):
    src = (hw_root / "robot_description/urdf/Robot.urdf").read_text()
    txt = src.replace("</robot>", urdf_block(variant, props, meshes) + "\n</robot>")
    txt = txt.replace('name="Robot"', 'name="Chhaya"', 1)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(txt)


def write_mjcf(variant, props, meshes, train_root, out_xml, mesh_rel):
    """The legs-only MJCF (robot_v2.xml) plus the chest, head and arm bodies = chhaya.xml."""
    src = train_root / "envs/assets/robotV2/robot_v2.xml"
    txt = src.read_text()
    # robot_v2.xml has "--" inside comments, which MuJoCo tolerates but strict XML does not
    txt = re.sub(r"<!--(.*?)-->", lambda m: "<!--" + m.group(1).replace("--", "-") + "-->", txt, flags=re.S)
    root = ET.fromstring(txt, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    tree = ET.ElementTree(root)
    root.set("model", "Chhaya")
    asset = root.find("asset")
    for link, mm in meshes.items():
        names = [link] + mm["cols"] + ([mm["servos"]] if mm["servos"] else [])
        for n in names:
            ET.SubElement(asset, "mesh", name=n, content_type="model/stl",
                          file=f"{mesh_rel}/{n}.stl", scale="0.001 0.001 0.001")
    base = root.find(".//body[@name='base_link']")
    bodies = {"base_link": base}
    sv = ub.SERVOS[variant]
    for link, (parent, joint, _, _) in links_for().items():
        M, com, I = props[link]
        o = (frame_origin_mm(link) - (frame_origin_mm(parent) if parent != "base_link" else 0)) / 1000
        b = ET.SubElement(bodies[parent], "body", name=link, pos=fmt(o))
        ET.SubElement(b, "inertial", pos=fmt(com), mass=f"{M:.6f}",
                      fullinertia=fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], 5))
        if joint is not None:
            lo, hi = LIMITS[joint[2:]]
            ET.SubElement(b, "joint", name=joint, pos="0 0 0", axis=fmt(ub.JOINTS[joint][1]),
                          range=f"{lo} {hi}", actuatorfrcrange=f"-{sv.stall_nm} {sv.stall_nm}")
        # visuals (no contacts)
        ET.SubElement(b, "geom", type="mesh", mesh=link, contype="0", conaffinity="0", group="1",
                      rgba="0.93 0.93 0.95 1")
        if meshes[link]["servos"]:
            ET.SubElement(b, "geom", type="mesh", mesh=meshes[link]["servos"], contype="0",
                          conaffinity="0", group="1", rgba="0.12 0.12 0.14 1")
        # collisions: one convex hull per printed part + a box per servo case
        for c in meshes[link]["cols"]:
            ET.SubElement(b, "geom", type="mesh", mesh=c, group="3", rgba="0.5 0.5 0.9 0.3", mass="0")
        for ctr, half in servo_boxes(variant, link):
            ET.SubElement(b, "geom", type="box", pos=fmt(ctr), size=fmt(half), group="3",
                          rgba="0.9 0.5 0.5 0.3", mass="0")
        bodies[link] = b
    # The shoulder and upper-arm links sit against the chest (their hulls overlap it), so those pairs are
    # excluded. The elbow link and the forearm (with the claw) DO collide with the torso and the chest: the
    # policy commands the arms, and its hands can reach the torso. Arm-vs-leg, arm-vs-arm and arm-vs-floor
    # contacts stay on too.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for s in "lr":
        for a in ("shoulder_link", "upper_arm_link"):
            ET.SubElement(contact, "exclude", body1="chest_link", body2=f"{s}_{a}")
            ET.SubElement(contact, "exclude", body1="base_link", body2=f"{s}_{a}")
        ET.SubElement(contact, "exclude", body1=f"{s}_shoulder_link", body2=f"{s}_elbow_link")
    act = root.find("actuator")
    if variant == "sts3215":
        act.append(ET.Comment(" Upper body: STS3215 torque pass-through, driven by the BAM model like the legs "))
        for j in UPPER_JOINT_ORDER:
            ET.SubElement(act, "motor", name=j, joint=j, ctrlrange=f"-{sv.stall_nm} {sv.stall_nm}")
    else:
        act.append(ET.Comment(" Upper body: MG90S hobby servos (no BAM model). Position servo with a "
                              "torque ceiling at the stall torque; randomise kp/kv in training. "))
        for j in UPPER_JOINT_ORDER:
            ET.SubElement(act, "position", name=j, joint=j, kp="3.0", kv="0.03",
                          forcerange=f"-{sv.stall_nm} {sv.stall_nm}", forcelimited="true",
                          ctrlrange=f"{LIMITS[j[2:]][0]} {LIMITS[j[2:]][1]}", ctrllimited="true")
        for j in UPPER_JOINT_ORDER:
            el = root.find(f".//joint[@name='{j}']")
            el.set("armature", "0.002")
            el.set("frictionloss", "0.01")
            el.set("damping", "0.005")
    ET.indent(tree, space="  ")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", type=Path, default=ROOT, help="hardware/ directory (default: this repository's)")
    ap.add_argument("--train", type=Path, default=ROOT.parent / "training", help="training/ directory (default: this repository's)")
    args = ap.parse_args()
    for variant in ("sts3215",):
        cad_dir = ub.CHECKS
        props, budget = mass_props(variant, cad_dir)
        rd = args.hw / "robot_description"
        md = rd / "meshes"                 # shared with the leg meshes: remove only our own files
        for link in links_for():
            for f in list(md.glob(f"{link}.stl")) + list(md.glob(f"{link}_servos.stl")) + list(md.glob(f"{link}_col*.stl")):
                f.unlink()
        meshes = write_meshes(variant, cad_dir, md)
        write_urdf(variant, props, meshes, args.hw, rd / "urdf" / "chhaya.urdf")
        write_mjcf(variant, props, meshes, args.train,
                   args.train / "envs/assets/robotV2" / "chhaya.xml",
                   "../../../robot_hardware/robot_description/meshes")
        upper = sum(p[0] for p in props.values())
        summary = {"variant": variant, "generated_links_kg": round(upper, 4),
                   "per_link_kg": {k: round(v[0], 4) for k, v in props.items()},
                   "items_g": budget}
        (rd / "chhaya_mass.json").write_text(json.dumps(summary, indent=2))
        print(f"chest + head + arms: {upper*1000:.0f} g", {k: round(v[0] * 1000) for k, v in props.items()})


if __name__ == "__main__":
    main()
