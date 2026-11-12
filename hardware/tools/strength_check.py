"""
Strength check of the printed parts (hand calculation on REAL cross-sections cut from the CAD).

For each critical section: cut the part with a thin slab, take the exact area and second moment
of area from the B-rep, apply the design loads and report the bending stress and the safety
factor against the printed strength in the direction the stress acts (set by how the part is
printed). Not a finite-element analysis: stress concentrations are covered by the required
safety factor (>= 3.0) and by the fillets on every inside corner.

    python tools/strength_check.py        -> cad/parametric/checks/strength.json, exits 1 below SF 3

Design loads
  T_STALL  2.94 N*m  STS3215 stall torque (what a joint can push with)
  T_IMPACT 4.9 N*m   peak joint torque in the simulated fall onto the arm (physics check T8)
  F_HAND   45 N      force on the claw tip, any direction (about 2x the robot's weight on one hand)
  F_HEAD   50 N      force on the top of the head hoop, fore-aft or sideways
  F_CHEST  75 N      inertia of the whole upper body (0.75 kg at 10 g) in a fall, fore-aft or sideways
Material (PLA; PETG is ~10 % lower but far tougher in falls)
  along the layers 50 MPa, across the layers 25 MPa
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "parametric"))
import cadquery as cq                       # noqa: E402
import chhaya_cad as ub                     # noqa: E402
from OCP.BRepGProp import BRepGProp         # noqa: E402
from OCP.GProp import GProp_GProps          # noqa: E402

CAD = ub.CHECKS
T_STALL, T_IMPACT, F_HAND, F_HEAD, F_CHEST = 2.94, 4.9, 45.0, 50.0, 75.0
S_ALONG, S_ACROSS = 50.0, 25.0
SF_MIN = 3.0
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def load(name):
    return cq.importers.importStep(str(ub.STEP_DIR / f"{name}.step"))


def section(part, normal, at, clip=None, t=0.2):
    """Area [mm2], centroid, and a function I(axis) [mm4] for the cut at `at` along `normal`."""
    lo, hi = np.full(3, -400.0), np.full(3, 400.0)
    k = "xyz".index(normal)
    lo[k], hi[k] = at - t / 2, at + t / 2
    if clip:
        for ax, (a, b) in clip.items():
            lo["xyz".index(ax)], hi["xyz".index(ax)] = a, b
    slab = part.intersect(ub.box(tuple(lo), tuple(hi)))
    shp = slab.val().wrapped
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shp, g)
    area = g.Mass() / t
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]) / t
    c = g.CentreOfMass()
    cen = np.array([c.X(), c.Y(), c.Z()])
    verts = np.array([[v.X, v.Y, v.Z] for v in slab.val().Vertices()])

    def I_c(axis):          # second moment about `axis` through the centroid, and the extreme fibre distance
        a = AX[axis]
        perp = [n for n in "xyz" if n not in (axis, normal)][0]
        cdist = np.abs(verts[:, "xyz".index(perp)] - cen["xyz".index(perp)]).max()
        return float(a @ I @ a), float(cdist)
    return area, cen, I_c


rows = []


def bend(name, part, normal, at, axis, M_nm, strength, how, clip=None):
    area, cen, I_c = section(part, normal, at, clip)
    I, c = I_c(axis)
    sigma = M_nm * 1000.0 * c / I
    rows.append({"where": name, "load": how, "area_mm2": round(area, 1), "I_mm4": round(I), "stress_MPa": round(sigma, 1),
                 "strength_MPa": strength, "SF": round(strength / sigma, 1)})


def direct(name, how, stress, strength):
    rows.append({"where": name, "load": how, "stress_MPa": round(stress, 1), "strength_MPa": strength,
                 "SF": round(strength / stress, 1)})


def main():
    rx, ry, rz = ub.ROLL
    ex, ey, ez = ub.ELBOW
    G = ub.gripper_geom()
    wx, wy, wz = ub.wrist_frame()
    px, py, pz = G["pivot"]
    tip_z = pz - 44.0
    ua, fa, cb, cj = load("l_upper_arm"), load("l_forearm"), load("l_claw_base"), load("l_claw_jaw")
    sb, el, hd = load("l_shoulder_bracket"), load("l_elbow_link"), load("head")

    # --- forks: the horn ear alone carries the joint torque, in its own plane (printed on its side: along layers)
    bend("upper-arm fork, horn ear at the window", ua, "z", rz - 24.0, "x", T_IMPACT, S_ALONG,
         "shoulder-roll torque 4.9 N*m", clip={"x": (rx - 0.1, rx + ub.EAR + 0.1)})
    bend("forearm fork, horn ear at the window", fa, "z", ez - 24.0, "y", T_IMPACT, S_ALONG,
         "elbow torque 4.9 N*m", clip={"y": (ey - 0.1, ey + ub.EAR + 0.1)})
    # --- forks: both ears together, bent sideways by a force on the hand
    bend("upper-arm fork, both ears, sideways", ua, "z", rz - 24.0, "y", T_IMPACT, S_ALONG, "4.9 N*m about Y")
    bend("forearm fork, both ears, sideways", fa, "z", ez - 24.0, "x", F_HAND * (ez - 24.0 - tip_z) / 1000, S_ALONG,
         "45 N sideways on the claw tip")
    # --- forearm shell (thin, open at the back), mid-height
    zs = (ez - ub.FORK_D - ub.FORK_TB + wz + ub.WRIST_T) / 2
    for axis, d in (("x", "sideways"), ("y", "fore-aft")):
        bend(f"forearm shell, mid-height, {d}", fa, "z", zs, axis, F_HAND * (zs - tip_z) / 1000, S_ALONG,
             f"45 N {d} on the claw tip")
    # --- claw module: housing walls beside the servo pocket (printed on its back: along layers)
    zp = (G["p_z0"] + G["p_z1"]) / 2
    for axis, d in (("x", "sideways"), ("y", "fore-aft")):
        bend(f"claw housing walls, {d}", cb, "z", zp, axis, F_HAND * (zp - tip_z) / 1000, S_ALONG,
             f"45 N {d} on the claw tip")
    # --- fixed jaw (finger), at its root and mid-way; and the moving jaw (limited by its servo)
    for dz, nm in ((-21.0, "root"), (-30.0, "middle")):
        for axis, d in (("x", "in the jaw plane"), ("y", "across the jaw plane")):
            bend(f"fixed jaw, {nm}, {d}", cb, "z", pz + dz, axis, F_HAND * (pz + dz - tip_z) / 1000, S_ALONG,
                 "45 N on the jaw tip", clip={"y": (py - 30, py - 1.0), "x": (G["jaw_x0"] - 12, G["jaw_x1"] + 1)})
    bend("moving jaw, near the hub", cj, "z", pz - 10.0, "x", 0.30, S_ALONG, "0.30 N*m (its micro servo stalls / strips first)")
    # --- wrist interface: 4x M3 self-tapping into the forearm plate
    xs, ys = ub.WRIST_SCREW_X, ub.WRIST_SCREW_Y
    a, b, _ = ub.WRIST_PLATE
    M = F_HAND * (wz - tip_z)                         # N*mm at the mating plane
    def pull(arm_far, arm_near):                      # tension in a far-row screw, pivoting on the plate edge
        return M * arm_far / (2 * (arm_far ** 2 + arm_near ** 2))
    t_fa = pull(a / 2 + xs, a / 2 - xs)
    t_side = pull(b / 2 + ys, b / 2 - ys)
    cap = math.pi * 3.0 * ub.WRIST_T * 0.75 * (S_ACROSS * 0.577)        # thread shear-out, 75 % engagement
    direct("wrist screws (4x M3), fore-aft", f"45 N on the claw tip -> {t_fa:.0f} N per screw", t_fa, cap)
    direct("wrist screws (4x M3), sideways", f"45 N on the claw tip -> {t_side:.0f} N per screw", t_side, cap)
    # --- brackets: face plate next to the horn pad, bent in its own plane by the next joint's torque
    y_cut = ub.SHOULDER[1] + ub.EAR + ub.FILLET + 1.0
    bend("shoulder bracket, plate beside the pad", sb, "y", y_cut, "x", T_IMPACT, S_ALONG, "shoulder-roll torque 4.9 N*m")
    # --- neck (column + rib) just above the flange; printed standing: stress crosses the layers
    zn = ub.CH_Z1 + 6.0
    lever = (ub.head_center()[2] + ub.HEAD_R_OUT - zn) / 1000
    bend("neck above the flange, fore-aft hit", hd, "z", zn, "y", F_HEAD * lever, S_ACROSS, "50 N on top of the hoop",
         clip={"z": (zn - 0.1, zn + 0.1)})
    bend("neck above the flange, side hit", hd, "z", zn, "x", F_HEAD * lever, S_ACROSS, "50 N on top of the hoop")
    # --- chest legs (front and back plates, printed standing: stress crosses the layers).
    # Fore-aft: each plate's two legs bend out of their plane like fixed-fixed columns.
    ch = load("chest_frame")
    z0, z1 = ub.FEET_Z + 3.0, ub.CH_Z0 - 2.0
    h = z1 - z0
    for zc, nm in ((z0 + 4.0, "just above the feet"), ((z0 + z1) / 2, "mid-height")):
        for (xa, xb), side in (((ub.CH_X1 - ub.T_CH - ub.LEG_RIB - 0.2, ub.CH_X1 + 0.05), "front"), ((ub.CH_X0 - 0.05, ub.CH_X0 + ub.T_CH + ub.LEG_RIB + 0.2), "back")):
            bend(f"chest legs, {side} plate, {nm}, fore-aft", ch, "z", zc, "y", (F_CHEST / 2) * (h / 2) / 1000, S_ACROSS,
                 "75 N fore-aft on the upper body", clip={"x": (xa, xb)})
    bend("chest legs, all four, sideways", ch, "z", (z0 + z1) / 2, "x", F_CHEST * (ub.CH_Z0 + 17 - (z0 + z1) / 2) / 1000, S_ACROSS,
         "75 N sideways on the upper body")
    # --- electronics rack: the chest stands on it. A fall tips the upper body (75 N at its centre of mass,
    # ~17 mm above the chest bottom) about one pair of feet; the far pair pulls up, the near pair pushes down.
    # Plate printed flat: bending in the plate's own layers. Section between each foot and its nearest support.
    rk = load("electronics_rack")
    h_cm = (ub.CH_Z0 + 17.0 - ub.FEET_Z) / 1000
    feet = ub.feet_points()
    span_x = (max(f[0] for f in feet) - min(f[0] for f in feet)) / 1000
    span_y = 2 * abs(feet[0][1]) / 1000
    f_foot = F_CHEST * h_cm / min(span_x, span_y) / 2          # per foot (two feet share the reaction)
    for (fx, fy) in [f for f in feet if f[1] > 0]:      # one front and one back foot (the plate is symmetric)
        bx, by = min(ub.BOSS_XY, key=lambda b: (b[0] - fx) ** 2 + (b[1] - fy) ** 2)
        xc = (fx + bx) / 2
        bend(f"rack plate, {'front' if fx > 0 else 'back'} foot to boss", rk, "x", xc, "y", f_foot * abs(xc - fx) / 1000, S_ALONG,
             f"75 N fall on the upper body -> {f_foot:.0f} N per foot", clip={"y": (min(fy, by) - 8.0, max(fy, by) + 8.0), "z": (ub.RACK_Z0 - 4.5, ub.FEET_Z + 0.1)})
    # rack screws: 4x M2.5 into the original bosses + 4x M3 through the torso plate (nuts) share the uplift
    cap_m25 = math.pi * 2.5 * 6.0 * 0.75 * (S_ACROSS * 0.577)   # thread shear-out in a boss, 6 mm engaged
    direct("rack screws into the original bosses (M2.5)", f"{f_foot:.0f} N uplift at a foot, one screw takes it", f_foot, cap_m25)

    # --- screws and bearing (closed form)
    sv = ub.STS3215
    f_horn = T_IMPACT * 1000 / (4 * sv.horn_pcd / 2)
    direct("horn screws (4x M3): bearing on the 4 mm ear", f"4.9 N*m -> {f_horn:.0f} N per screw", f_horn / (3.0 * ub.EAR), S_ALONG)
    r_face = math.hypot(12.25, 10.25)
    f_face = T_IMPACT * 1000 / (4 * r_face)
    direct("face screws (4x M2): bearing on the 3 mm plate", f"4.9 N*m -> {f_face:.0f} N per screw", f_face / (2.0 * ub.PLATE), S_ALONG)
    m_stub = (F_HAND / 2) * (ub.IDLER_GAP + ub.EAR / 2)
    direct("idler stub (dia 8), bending at the collar", "half of 45 N at the ear", m_stub * ub.STUB_R / (math.pi * ub.STUB_R ** 4 / 4), S_ACROSS)

    ok = True
    print(f"{'section':<46}{'stress':>8}{'allow':>7}{'SF':>6}   load")
    for r in rows:
        flag = "" if r["SF"] >= SF_MIN else "   <-- below 3"
        ok &= r["SF"] >= SF_MIN
        print(f"{r['where']:<46}{r['stress_MPa']:>7.1f} {r['strength_MPa']:>6.0f}{r['SF']:>6.1f}   {r['load']}{flag}")
    (CAD / "strength.json").write_text(json.dumps({"loads": {"T_STALL": T_STALL, "T_IMPACT": T_IMPACT, "F_HAND": F_HAND, "F_HEAD": F_HEAD},
                                                    "strength_MPa": {"along_layers": S_ALONG, "across_layers": S_ACROSS},
                                                    "sf_min": SF_MIN, "sections": rows}, indent=1))
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
