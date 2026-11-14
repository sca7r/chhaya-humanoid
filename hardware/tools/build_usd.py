"""
Build the Isaac Lab USD : envs/assets/robotV2/chhaya.usd.

No Isaac Sim needed (pip install usd-core). The new file is a thin layer that SUBLAYERS the
existing robot.usd, so the legs stay byte-for-byte what the current policies were trained on,
and adds the chest, head and arms with the same authoring conventions as Isaac Sim's URDF importer:

  * links are Xforms directly under /World/Robot with RigidBodyAPI + MassAPI, placed at their
    zero-pose world transform; joints live in /World/Robot/joints;
  * revolute joints use axis X/Y/Z, and a negative URDF axis is expressed by rotating both
    joint frames 180 deg (localRot0 = localRot1), exactly as the importer does for r_hip_yaw;
  * fixed joints are merged into their parent (as the importer did with imu_link): the chest
    and head become part of base_link, whose mass properties are updated to include them;
  * visuals under <link>/visuals, colliders under <link>/collisions (convex hulls + a box per
    servo case), with FilteredPairsAPI on the same pairs the MJCF excludes.

    python tools/build_usd.py --train <BipedRobot checkout>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "parametric"))
sys.path.insert(0, str(ROOT / "tools"))
import chhaya_cad as ub                                                 # noqa: E402
from build_description import LIMITS, links_for, mass_props, frame_origin_mm, combine   # noqa: E402

VARIANT = "sts3215"
CAD = ub.CHECKS
MESHES = ROOT / "robot_description/meshes"

# importer convention for negative axes: (w, x, y, z) quaternion that maps the token axis onto -axis
FLIP = {"X": (0.0, 0.0, 0.0, 1.0),      # 180 deg about Z: X -> -X
        "Y": (0.0, 1.0, 0.0, 0.0),      # 180 deg about X: Y -> -Y
        "Z": (0.0, 0.0, 1.0, 0.0)}      # 180 deg about Y: Z -> -Z


def axis_token(axis):
    i = int(np.argmax(np.abs(axis)))
    return "XYZ"[i], (1 if axis[i] > 0 else -1)


def principal(I):
    """Full inertia tensor -> (diagonal, principal-axes quaternion (w,x,y,z))."""
    w, R = np.linalg.eigh(I)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()   # rows of Gf = columns of R
    return w, q


def add_mesh(stage, path, mesh_file, purpose="default", collider=False):
    m = trimesh.load(mesh_file)
    m.apply_scale(0.001)                       # files are in mm
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in m.vertices.tolist()]))
    g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(m.faces)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray(m.faces.flatten().tolist()))
    g.CreateSubdivisionSchemeAttr("none")
    g.CreatePurposeAttr(purpose)
    if collider:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim())
        mc.CreateApproximationAttr("convexHull")
        UsdGeom.Imageable(g).MakeInvisible()
    else:
        g.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.92, 0.95) if "servos" not in str(mesh_file) else Gf.Vec3f(0.12, 0.12, 0.14)])
    return g


def add_box(stage, path, center_m, half_m):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*center_m))
    x.AddScaleOp().Set(Gf.Vec3f(*(2 * np.asarray(half_m)).tolist()))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdGeom.Imageable(c).MakeInvisible()
    return c


def read_base_inertial(urdf_text):
    blk = re.search(r'<link name="base_link">.*?<inertial>(.*?)</inertial>', urdf_text, re.S).group(1)
    xyz = [float(v) for v in re.search(r'origin xyz="([^"]+)"', blk).group(1).split()]
    m = float(re.search(r'mass value="([^"]+)"', blk).group(1))
    g = {k: float(v) for k, v in re.findall(r'(i[xyz]{2})="([^"]+)"', blk)}
    I = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
    return m, np.array(xyz), I


def build(train_root: Path, out: Path):
    props, _ = mass_props(VARIANT, CAD)
    info = json.loads((CAD / "parts.json").read_text())
    urdf = (train_root / "robot_hardware/robot_description/urdf/Robot.urdf").read_text()

    layer = Sdf.Layer.CreateNew(str(out))
    layer.subLayerPaths.append("./robot.usd")
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetEditTarget(layer)

    # ---- chest (+ head) merged into base_link ---------------------------------------------
    mb, cb, Ib = read_base_inertial(urdf)
    Mc, cc, Ic = props["chest_link"]
    M, com, I = combine([(mb, cb, Ib), (Mc, cc, Ic)])
    diag, q = principal(I)
    base = stage.GetPrimAtPath("/World/Robot/base_link")
    mapi = UsdPhysics.MassAPI(base)
    mapi.GetMassAttr().Set(float(M))
    mapi.GetCenterOfMassAttr().Set(Gf.Vec3f(*com.tolist()))
    mapi.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diag.tolist()))
    mapi.GetPrincipalAxesAttr().Set(Gf.Quatf(q))
    UsdGeom.Xform.Define(stage, "/World/Robot/base_link/upper_visuals")
    add_mesh(stage, "/World/Robot/base_link/upper_visuals/chest", MESHES / "chest_link.stl")
    add_mesh(stage, "/World/Robot/base_link/upper_visuals/chest_servos", MESHES / "chest_link_servos.stl")
    UsdGeom.Xform.Define(stage, "/World/Robot/base_link/upper_collisions")
    meshes = sorted(MESHES.glob("chest_link_col*.stl"))
    for i, f in enumerate(meshes):
        add_mesh(stage, f"/World/Robot/base_link/upper_collisions/hull{i}", f, collider=True)
    for i, s in enumerate(links_for()["chest_link"][3]):
        lo, hi = (np.array(b) / 1000 for b in info["servos"][s]["case_bounds_mm"])
        add_box(stage, f"/World/Robot/base_link/upper_collisions/servo{i}", (lo + hi) / 2, (hi - lo) / 2)

    # ---- arm links and joints -----------------------------------------------------------
    for link, (parent, joint, _, servos) in links_for().items():
        if joint is None:
            continue
        o = frame_origin_mm(link) / 1000
        xf = UsdGeom.Xform.Define(stage, f"/World/Robot/{link}")
        # same op stack and precision as the importer's links (translate, orient, scale; doubles)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*o.tolist()))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        Mi, ci, Ii = props[link]
        d, qi = principal(Ii)
        ma = UsdPhysics.MassAPI.Apply(prim)
        ma.CreateMassAttr(float(Mi))
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*ci.tolist()))
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*d.tolist()))
        ma.CreatePrincipalAxesAttr(Gf.Quatf(qi))
        UsdGeom.Xform.Define(stage, f"/World/Robot/{link}/visuals")
        add_mesh(stage, f"/World/Robot/{link}/visuals/parts", MESHES / f"{link}.stl")
        if servos:
            add_mesh(stage, f"/World/Robot/{link}/visuals/servos", MESHES / f"{link}_servos.stl")
        UsdGeom.Xform.Define(stage, f"/World/Robot/{link}/collisions")
        for i, f in enumerate(sorted(MESHES.glob(f"{link}_col*.stl"))):
            add_mesh(stage, f"/World/Robot/{link}/collisions/hull{i}", f, collider=True)
        for i, s in enumerate(servos):
            lo, hi = (np.array(b) / 1000 for b in info["servos"][s]["case_bounds_mm"])
            add_box(stage, f"/World/Robot/{link}/collisions/servo{i}", (lo + hi) / 2 - o, (hi - lo) / 2)

        par = "base_link" if parent == "chest_link" else parent
        po = np.zeros(3) if parent == "chest_link" else frame_origin_mm(parent) / 1000
        tok, sgn = axis_token(ub.JOINTS[joint][1])
        rot = (1.0, 0.0, 0.0, 0.0) if sgn > 0 else FLIP[tok]
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/joints/{joint}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/Robot/{par}")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/Robot/{link}")])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(o - po).tolist()))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(*rot))
        j.CreateLocalRot1Attr(Gf.Quatf(*rot))
        j.CreateAxisAttr(tok)
        lo, hi = LIMITS[joint[2:]]
        j.CreateLowerLimitAttr(math.degrees(lo))
        j.CreateUpperLimitAttr(math.degrees(hi))
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(0.0)           # explicit BAM actuator: Isaac Lab zeroes the drive anyway
        drv.CreateDampingAttr(0.0)
        drv.CreateMaxForceAttr(100.0)
        drv.CreateTargetPositionAttr(0.0)

    # ---- collision filtering: same pairs the MJCF excludes -----------------------------------
    arm = [f"{s}_{a}" for s in "lr" for a in ("shoulder_link", "upper_arm_link")]
    fp = UsdPhysics.FilteredPairsAPI.Apply(base)
    fp.CreateFilteredPairsRel().SetTargets([Sdf.Path(f"/World/Robot/{a}") for a in arm])
    for s in "lr":
        fps = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"/World/Robot/{s}_shoulder_link"))
        fps.CreateFilteredPairsRel().SetTargets([Sdf.Path(f"/World/Robot/{s}_elbow_link")])
    layer.Save()
    return out


def verify(path: Path):
    """Composition and consistency checks (no physics): bodies, joints, masses, joint frames."""
    st = Usd.Stage.Open(str(path))
    bodies = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    mass = sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies)
    cache = UsdGeom.XformCache()
    worst = 0.0
    for jp in joints:
        j = UsdPhysics.RevoluteJoint(jp)
        b0 = st.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
        b1 = st.GetPrimAtPath(j.GetBody1Rel().GetTargets()[0])
        w0 = cache.GetLocalToWorldTransform(b0)
        w1 = cache.GetLocalToWorldTransform(b1)
        p0 = w0.Transform(Gf.Vec3d(*j.GetLocalPos0Attr().Get()))
        p1 = w1.Transform(Gf.Vec3d(*j.GetLocalPos1Attr().Get()))
        worst = max(worst, (p0 - p1).GetLength())
    names = [p.GetName() for p in joints]
    return {"bodies": len(bodies), "revolute_joints": len(joints), "total_mass_kg": round(mass, 4),
            "joint_frame_mismatch_mm": round(worst * 1000, 4), "arm_joints": [n for n in names if "shoulder" in n or "arm" in n or "elbow" in n]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT.parent / "training")
    a = ap.parse_args()
    out = a.train / "envs/assets/robotV2/chhaya.usd"
    if out.exists():
        out.unlink()
    build(a.train, out)
    print(out, json.dumps(verify(out), indent=1))
    # Isaac Lab 3.0 / Newton loads the converted copy (colliders on the meshes, convex hulls, no MDL).
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_newton_usd", out.parent / "make_newton_usd.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    newton = out.with_name("chhaya_newton.usd")
    print(newton, mod.convert(str(out), str(newton)))


if __name__ == "__main__":
    main()
