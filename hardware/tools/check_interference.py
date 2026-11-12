"""
Kinematic interference and clearance check for the chest, head, arms and claws (no physics). Exact mesh-vs-mesh
tests (FCL via trimesh), not convex hulls.

1. Interference: moves the CAD parts (and servo envelopes incl. horns) through every joint's
   range plus random whole-arm poses and fails on ANY overlap between parts of different links.
2. Torso fit (exact B-rep, no tessellation): the original body, the electronics rack, the chest frame,
   the battery and the computer must not overlap (touching = seated is fine), and the battery and the
   computer must clear the chest and the body by >= CLR_MOVE (they are dropped in and strapped).
3. Clearance: sweeps each arm joint and reports the smallest gap between everything on the
   moving side and everything on the fixed side, measured against the bare servo CASES (a part
   screwed to a horn is supposed to touch that horn). Every gap must be >= CLR_MOVE (1.0 mm,
   enough for print tolerance and elephant's foot), except the idler bearings (stub in ear
   bore), which are a deliberate running fit.

    python tools/check_interference.py [--random 80]

Writes cad/parametric/checks/interference.json and clearance.json; exits 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "parametric"))
sys.path.insert(0, str(ROOT / "tools"))
import chhaya_cad as ub                       # noqa: E402
from build_description import LIMITS, links_for   # noqa: E402

CAD = ub.CHECKS
TESS = 0.05          # mesh tessellation slack [mm]


def rot(axis, q):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def link_transforms(q):
    """4x4 transform of every link (applied to geometry modelled at the zero pose)."""
    T = {"base_link": np.eye(4), "chest_link": np.eye(4)}
    for link, (parent, joint, _, _) in links_for().items():
        if joint is None:
            continue
        o, ax = ub.JOINTS[joint]
        o = np.array(o, float)
        M = np.eye(4)
        M[:3, :3] = rot(ax, q.get(joint, 0.0))
        M[:3, 3] = o - M[:3, :3] @ o
        T[link] = T[parent] @ M
    return T


def case_mesh(servo_name):
    """Bare servo case (no horn) as a box mesh, from the same placement the CAD uses."""
    side, kind = servo_name[0], servo_name[2:]
    p = ub.placements(ub.STS3215, side)[kind]
    lo, hi = (np.array(v) for v in p.case_bounds())
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def load_objects(cases_only=False):
    objs = []   # (name, link, mesh)
    for link, (_, _, parts, servos) in links_for().items():
        for p in parts:
            objs.append((p, link, trimesh.load(ub.STL_DIR / f"{p}.stl")))
        for s in servos:
            m = case_mesh(s) if cases_only else trimesh.load(CAD / "servo_standins" / f"{s}.stl")
            objs.append((f"servo_{s}", link, m))
    info = json.loads((CAD / "parts.json").read_text())
    for n, e in info.get("extras", {}).items():
        objs.append((n, e["link"], trimesh.load(CAD / "extras" / f"{n}.stl")))
    return objs


def check(q, objs):
    """Overlapping pairs of different links at pose q -> {(a, b): depth}."""
    T = link_transforms(q)
    cm = trimesh.collision.CollisionManager()
    owner = {}
    for name, link, mesh in objs:
        cm.add_object(name, mesh, transform=T[link])
        owner[name] = link
    _, contacts = cm.in_collision_internal(return_names=False, return_data=True)
    bad = {}
    for c in contacts:
        a, b = sorted(c.names)
        if owner[a] != owner[b] and c.depth > TESS:   # depth ~0 = flush seat on its own horn
            bad[(a, b)] = max(bad.get((a, b), 0.0), float(c.depth))
    return bad


def is_bearing(a, b):
    names = {a, b}
    return any("idler_cap" in n for n in names) and any(n.endswith(("upper_arm", "forearm")) for n in names)


def clearances(objs, n=31):
    """For each left-arm joint: min gap between the joint's child link and its parent link
    (plus the chest) over the joint's range."""
    by_link = {}
    for name, link, mesh in objs:
        by_link.setdefault(link, []).append((name, mesh))
    out = {}
    for child, (parent, joint, _, _) in links_for().items():
        if joint is None or not joint.startswith("l_"):
            continue
        fixed = {parent, "chest_link"}
        lo, hi = LIMITS[joint[2:]]
        for v in np.linspace(lo, hi, n):
            T = link_transforms({joint: float(v)})
            for an, am in by_link[child]:
                for fl in fixed:
                    for bn, bm in by_link.get(fl, []):
                        cm = trimesh.collision.CollisionManager()
                        cm.add_object("a", am, transform=T[child])
                        d = float(cm.min_distance_single(bm, transform=T[fl]))
                        key = f"{an} | {bn}"
                        if key not in out or d < out[key]["gap_mm"]:
                            out[key] = {"joint": joint, "at_rad": round(float(v), 2), "gap_mm": round(d, 2),
                                        "bearing": is_bearing(an, bn)}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["gap_mm"]))


def claw_sweep():
    """The claw is not a simulated joint, so sweep it here: the moving jaw from just short of
    closed (-8 deg, where the tips are ~1 mm apart) to fully open (+60 deg) against the claw
    base, its micro servo and the forearm. Returns the smallest gap [mm] (negative = overlap)."""
    G = ub.gripper_geom()
    piv = np.array([0.0, G["pivot"][1], G["pivot"][2]])
    jaw = trimesh.load(ub.STL_DIR / "l_claw_jaw.stl")
    others = {"claw base": trimesh.load(ub.STL_DIR / "l_claw_base.stl"),
              "micro servo": trimesh.load(CAD / "extras" / "l_claw_servo.stl"),
              "forearm": trimesh.load(ub.STL_DIR / "l_forearm.stl")}
    out = {}
    for deg in range(-8, 61, 4):
        a = np.radians(deg)
        T = np.eye(4)
        T[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        T[:3, 3] = piv - T[:3, :3] @ piv
        cm = trimesh.collision.CollisionManager()
        cm.add_object("jaw", jaw, transform=T)
        for n, m in others.items():
            d = -1.0 if cm.in_collision_single(m) else float(cm.min_distance_single(m))
            if n not in out or d < out[n][0]:
                out[n] = (round(d, 2), deg)
    return out


FIT_SEATED_MM3 = 1.0     # a face-to-face seat leaves a zero-thickness boolean sliver; anything bigger is an overlap


def torso_fit():
    """Overlap [mm3] and gap [mm] between every pair of fixed torso parts. Returns (rows, ok)."""
    import cadquery as cq
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    step = lambda n: cq.importers.importStep(str(ub.STEP_DIR / f"{n}.step")).val()
    objs = {"body (original)": step("body"), "electronics_rack": step("electronics_rack"),
            "chest_frame": step("chest_frame"), "battery": ub.battery_standin().val(), "computer": ub.pi_standin().val()}
    need_gap = {frozenset(p) for p in (("battery", "chest_frame"), ("computer", "chest_frame"),
                                       ("battery", "body (original)"), ("computer", "body (original)"),
                                       ("battery", "computer"))}
    names, rows, ok = list(objs), [], True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = objs[names[i]], objs[names[j]]
            ov = a.intersect(b).Volume()
            d = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
            d.Perform()
            gap = d.Value()
            good = ov <= FIT_SEATED_MM3 and (frozenset((names[i], names[j])) not in need_gap or gap >= ub.CLR_MOVE)
            ok &= good
            rows.append({"a": names[i], "b": names[j], "overlap_mm3": round(ov, 2), "gap_mm": round(gap, 2), "ok": good})
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=80)
    args = ap.parse_args()
    ok = True

    objs = load_objects()
    joints = list(ub.JOINTS)
    poses = [("zero", {})]
    for j in joints:
        lo, hi = LIMITS[j[2:]]
        for v in np.linspace(lo, hi, 9):
            poses.append((f"{j}={v:+.2f}", {j: float(v)}))
    rng = np.random.default_rng(0)
    for k in range(args.random):
        poses.append((f"random{k}", {j: float(rng.uniform(*LIMITS[j[2:]])) for j in joints}))
    report = {}
    for name, q in poses:
        bad = check(q, objs)
        if bad:
            report[name] = {f"{a} x {b}": round(d, 3) for (a, b), d in bad.items()}
    (CAD / "interference.json").write_text(json.dumps(report, indent=2))
    print(f"interference: {len(poses)} poses, {len(report)} with any overlap")
    for k, v in list(report.items())[:20]:
        print("  ", k, v)
    ok &= not report

    rows, fit_ok = torso_fit()
    (CAD / "torso_fit.json").write_text(json.dumps(rows, indent=2))
    print("\ntorso fit (original body, rack, chest, battery, computer):")
    for r in rows:
        print(f"  {r['a']:<17} x {r['b']:<17} overlap {r['overlap_mm3']:6.2f} mm3  gap {r['gap_mm']:6.2f} mm  {'OK' if r['ok'] else 'FAIL'}")
    ok &= fit_ok

    cl = clearances(load_objects(cases_only=True))
    (CAD / "clearance.json").write_text(json.dumps(cl, indent=2))
    need = ub.CLR_MOVE - TESS
    print(f"\nclearance (moving side vs fixed side, bare servo cases; need >= {ub.CLR_MOVE} mm):")
    for k, v in cl.items():
        if v["gap_mm"] > 6:
            continue
        flag = "bearing (running fit)" if v["bearing"] else ("OK" if v["gap_mm"] >= need else "TOO TIGHT")
        print(f"  {v['gap_mm']:6.2f} mm  {v['joint']:<17} at {v['at_rad']:+.2f}  {k:<52} {flag}")
        if not v["bearing"] and v["gap_mm"] < need:
            ok = False
    print("\nclaw jaw sweep, -8 deg (tips 1 mm apart) to +60 deg (open):")
    for n, (d, deg) in claw_sweep().items():
        print(f"  {d:6.2f} mm  at {deg:+d} deg  moving jaw | {n:<12} {'OK' if d >= need else 'TOO TIGHT'}")
        ok &= d >= need
    cl["claw_sweep"] = {n: {"gap_mm": d, "at_deg": deg} for n, (d, deg) in claw_sweep().items()}
    (CAD / "clearance.json").write_text(json.dumps(cl, indent=2))
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
