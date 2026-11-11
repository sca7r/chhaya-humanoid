"""
Static check of the generated robot description (no simulation): compiles the MJCF and the URDF
in MuJoCo, and prints the joint table, actuators, masses and centre of mass.

    python tools/validate_model.py
"""
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path(__file__).resolve().parents[3] / "training")
    a = ap.parse_args()
    xml = a.train / "envs/assets/robotV2/robot_v2_upper_sts3215.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"MuJoCo {mujoco.__version__}: {xml.name} compiles -- nq={m.nq} nv={m.nv} nu={m.nu} bodies={m.nbody}")
    print(f"total mass {m.body_subtreemass[1]:.3f} kg (legs-only robot_v2: 1.661 kg)")
    com = d.subtree_com[1] - d.xpos[1]
    print(f"CoM relative to base_link origin: {np.round(com * 1000, 1)} mm")
    print("\njoint                 range [rad]          actuator")
    for j in range(1, m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        act = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu) if m.actuator_trnid[i, 0] == j]
        print(f"  {name:20s} [{m.jnt_range[j, 0]:+.3f}, {m.jnt_range[j, 1]:+.3f}]   {act[0] if act else '-'}")
    print("\nupper-body links:")
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if any(k in n for k in ("chest", "shoulder", "upper_arm", "elbow", "forearm")):
            print(f"  {n:20s} {m.body_mass[b] * 1000:6.1f} g")
    # URDF: resolve package:// paths and drop the xacro includes, then compile
    hw = a.train / "robot_hardware/robot_description"
    t = (hw / "urdf/Robot_upper_sts3215.urdf").read_text()
    t = re.sub(r"<xacro:include[^>]*>(</xacro:include>)?", "", t).replace('xmlns:xacro="http://www.ros.org/wiki/xacro"', "")
    t = t.replace("package://Robot_description/", str(hw) + "/")
    t = t.replace("</robot>", '<mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco></robot>')
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(t)
    mu = mujoco.MjModel.from_xml_path(f.name)
    print(f"\nURDF compiles in MuJoCo: {mu.njnt} joints, mass {sum(mu.body_mass):.3f} kg")


if __name__ == "__main__":
    main()
