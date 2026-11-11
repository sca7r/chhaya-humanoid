"""
Parametric upper body for the BipedRobot ("Sir Servington") -- CadQuery.

Design language: the same as the robot's legs (thigh / calf / hip_and_ankle):
  * the servos are the structure; printed parts are flat 3.5-4 mm plates that screw to the
    servos' own M2 face holes, exactly like the leg brackets do;
  * dual-support FORKS at the roll and elbow joints: one ear on the horn, the other on a
    printed idler stub screwed to the servo's back face, so the joint load goes into both sides
    of the servo instead of cantilevering off the horn;
  * large concave fillets wherever two plates meet, rounded ears, pocketed cut-outs: no sharp
    inside corners (stress raisers) and no material that carries no load;
  * modular: the upper-arm fork and the forearm fork are the same geometry, and one idler cap
    serves both forked joints.

Kinematics (left arm, base_link frame, mm; the right arm mirrors Y), arms hanging at zero:
  l_shoulder_pitch  axis +Y                 servo held in the chest
  l_shoulder_roll   axis +X                 servo held by the shoulder bracket
  l_arm_yaw         axis +Z (horn down)     servo held by the upper-arm fork bridge
  l_elbow           axis +Y                 servo held by the elbow link
The head is a fixed open ring with a ring of M3 holes, so anything can be mounted on it.

STS3215 face holes (4x M2 on the horn face and on the back face, 24.5 x 20.5 mm, at 8.2 and
32.7 mm from the shaft axis) were measured from the screw positions in the reference robot's
assembly STEP.

Run:  python cad/upper_body.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


# ------------------------------------------------------------------------------------------
# Servo
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Servo:
    name: str
    L: float = 45.22        # case length (along l)
    W: float = 24.72        # case width  (along w)
    H: float = 35.0         # case height, back face -> horn-side face
    a: float = 10.1         # shaft axis -> near case end
    horn_t: float = 4.0     # horn-side face -> horn outer face
    horn_r: float = 10.5
    horn_pcd: float = 14.0  # horn screw pitch circle -- CHECK against your horn
    horn_hole: float = 3.2  # M3 clearance (datasheet: horn screws M3)
    face_holes_l: tuple = (8.2, 32.7)
    face_holes_w: float = 10.25
    mass_kg: float = 0.060  # servo + horn + screws
    stall_nm: float = 2.94


STS3215 = Servo("sts3215")
SERVOS = {"sts3215": STS3215}

# ------------------------------------------------------------------------------------------
# Layout (mm, base_link frame)
# ------------------------------------------------------------------------------------------
SHELF_TOP_Z = 70.1
SHOULDER = (10.0, 66.0, 122.0)    # shoulder-pitch horn face centre
ROLL = (29.5, 95.0, 122.0)        # shoulder-roll horn face centre (faces +X)
YAW = (19.0, 95.0, 40.0)          # arm-yaw horn face centre (faces -Z)
ELBOW = (19.0, 116.5, 20.0)       # elbow horn face centre (faces +Y)
FOREARM_TO_TIP = 125.0

JOINTS = {
    "l_shoulder_pitch": (SHOULDER, (0, 1, 0)),
    "l_shoulder_roll": (ROLL, (1, 0, 0)),
    "l_arm_yaw": (YAW, (0, 0, 1)),
    "l_elbow": (ELBOW, (0, 1, 0)),
}
# right arm: roll and yaw axes are mirrored so that +roll = arm out and +yaw = upper arm outward
# on both sides; pitch and elbow keep +Y on both sides (like the legs' pitch joints)
for _k, (_p, _a) in list(JOINTS.items()):
    _m = _a if _k.endswith(("pitch", "elbow")) else tuple(-c for c in _a)
    JOINTS["r" + _k[1:]] = ((_p[0], -_p[1], _p[2]), _m)

PLATE = 3.0            # face plates (case top -> horn face is 4 mm: leaves 1.0 mm under the moving ear)
EAR = 4.0              # fork ears / horn pads
EAR_R = 17.0           # ear radius around the joint axis
CAP_T = 2.5            # idler cap plate
COLLAR_H = 1.0         # filleted collar at the stub root
IDLER_GAP = COLLAR_H + 1.0   # idler ear inner face -> cap outer face (1 mm clear of the collar)
STUB_L = IDLER_GAP + 4.0 + 0.3   # through the 4 mm ear + 0.3 so the washer clamps the stub, not the ear
STUB_R = 4.0           # idler stub radius (ear bore 8.4)
HOLE_M2 = 2.2
CB_M2 = (4.2, 1.6)     # counterbore for M2 heads (dia, depth): keeps heads below the moving ear
FILLET = 6.0           # concave fillet where two plates meet
CLR = 0.3              # around a servo a part is screwed to
CLR_MOVE = 1.0         # around a servo a part moves past (print tolerance + elephant's foot)

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

# look of the parts (concept variants): outline oct|round, edge chamfer|fillet, forearm truss|shell
CORNER_R = 5.0           # fork outline: bridge-corner radius
STYLE = {"outline": "round", "edge": "fillet", "win_r": 2.5, "forearm": "shell", "jaw_r": 2.0, "house": "round"}


def V(*a):
    return cq.Vector(*a)


def neg(v):
    return tuple(-c for c in v)


def add(p, *vs):
    return tuple(p[i] + sum(v[i] for v in vs) for i in range(3))


def mul(v, k):
    return tuple(c * k for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def box(lo, hi):
    lo2 = [min(a, b) for a, b in zip(lo, hi)]
    hi2 = [max(a, b) for a, b in zip(lo, hi)]
    return (cq.Workplane("XY").box(*[h - l for l, h in zip(lo2, hi2)])
            .translate(tuple((l + h) / 2 for l, h in zip(lo2, hi2))))


def cyl(center, axis, r, h0, h1):
    ax = V(*axis)
    return cq.Workplane(cq.Solid.makeCylinder(r, h1 - h0, V(*center) + ax * h0, ax))


def prism(origin, xdir, ydir, sketch, n0, n1):
    """Extrude a 2D sketch (local x along xdir, y along ydir, from origin) between offsets n0
    and n1 along the normal xdir x ydir."""
    nrm = cross(xdir, ydir)
    pl = cq.Plane(origin=origin, xDir=xdir, normal=nrm)
    return cq.Workplane(pl).workplane(offset=n0).placeSketch(sketch).extrude(n1 - n0)


def poly_sketch(pts, r=0.0):
    sk = cq.Sketch().polygon(list(pts) + [pts[0]])
    if r > 0:
        sk = sk.vertices().fillet(r)
    return sk


def rrect_sketch(x0, x1, y0, y1, r):
    return (cq.Sketch().push([((x0 + x1) / 2, (y0 + y1) / 2)]).rect(x1 - x0, y1 - y0)
            .reset().vertices().fillet(r))


def fillet_strip(corner, a_dir, b_dir, along, r, l0, l1):
    """Concave fillet in the inside corner between two plates. corner: a point on the corner
    line; a_dir / b_dir: unit vectors from the corner along each plate's free face (into the
    open space); along: corner line direction; spans [l0, l1] along it."""
    n = V(*along)
    a = V(*a_dir)
    o = V(*corner) + n * l0
    pl = cq.Plane(origin=o.toTuple(), xDir=a_dir, normal=along)
    sgn = 1.0 if n.cross(a).dot(V(*b_dir)) > 0 else -1.0
    blk = cq.Workplane(pl).center(r / 2, sgn * r / 2).rect(r, r).extrude(l1 - l0)
    cut = cq.Workplane(pl).center(r, sgn * r).circle(r).extrude(l1 - l0)
    return blk.cut(cut)


def _vol(wp):
    try:
        v = wp.val()
        return v.Volume() if v.isValid() else -1.0
    except Exception:
        return -1.0


def solidify(wp):
    shp = wp.val()
    sols = shp.Solids() if hasattr(shp, "Solids") else [shp]
    if len(sols) == 1:
        return cq.Workplane(sols[0].clean())
    return cq.Workplane(sols[0].fuse(*sols[1:]).clean())


def U(*items):
    """Checked union (retries with fuzzy tolerance on OpenCascade mis-fuses)."""
    out = None
    for it in items:
        if it is None:
            continue
        if out is None:
            out = solidify(it)
            continue
        need = 0.995 * max(_vol(out), _vol(it))
        for tol in (None, 1e-3, 1e-2):
            try:
                r = solidify(out.union(it, tol=tol) if tol else out.union(it))
            except Exception:
                continue
            if _vol(r) >= need:
                out = r
                break
        else:
            raise RuntimeError("union failed")
    return out


def C(part, *tools):
    """Checked cut: the result must be valid and no bigger than the input."""
    for t in tools:
        if t is None:
            continue
        vp = _vol(part)
        for tol in (None, 1e-3, 1e-2):
            try:
                r = solidify(part.cut(t, tol=tol) if tol else part.cut(t))
            except Exception:
                continue
            if 0.2 * vp < _vol(r) <= vp + 1e-6:
                part = r
                break
        else:
            raise RuntimeError("cut failed")
    return part


# ------------------------------------------------------------------------------------------
# Servo placement and its standard interfaces
# ------------------------------------------------------------------------------------------
@dataclass
class Placed:
    """Servo with its horn face centre at P, shaft direction s, length direction l (near case
    end -> shaft -> far end); w = s x l."""
    servo: Servo
    P: tuple
    s: tuple
    l: tuple

    @property
    def w(self):
        return cross(self.s, self.l)

    def at(self, s=0.0, l=0.0, w=0.0):
        return add(self.P, mul(self.s, s), mul(self.l, l), mul(self.w, w))

    def aabb(self, s0, s1, l0, l1, w0, w1):
        pts = [self.at(a, b, c) for a in (s0, s1) for b in (l0, l1) for c in (w0, w1)]
        return (tuple(min(p[i] for p in pts) for i in range(3)),
                tuple(max(p[i] for p in pts) for i in range(3)))

    @property
    def s_top(self):      # horn-side case face
        return -self.servo.horn_t

    @property
    def s_back(self):     # back case face
        return -(self.servo.horn_t + self.servo.H)

    def case_bounds(self, grow=0.0):
        sv = self.servo
        return self.aabb(self.s_back - grow, self.s_top + grow, -sv.a - grow, sv.L - sv.a + grow,
                         -sv.W / 2 - grow, sv.W / 2 + grow)

    def solid(self):
        sv = self.servo
        return box(*self.case_bounds()).union(cyl(self.P, self.s, sv.horn_r, -sv.horn_t, 0))

    def keepout(self, clr=CLR, mate=None):
        """Case + horn, grown by `clr`, for cutting out of brackets. mate="top"/"back": the part
        is screwed flat onto that case face, so the box stops 0.05 short of it there (no gap
        under the plate). mate="horn": the part is screwed onto the horn, so the horn cut stops
        0.05 short of the horn face (the part seats on the horn, it doesn't float above it)."""
        sv = self.servo
        g_top = -0.05 if mate == "top" else clr
        g_back = -0.05 if mate == "back" else clr
        b = self.aabb(self.s_back - g_back, self.s_top + g_top, -sv.a - clr, sv.L - sv.a + clr,
                      -sv.W / 2 - clr, sv.W / 2 + clr)
        return box(*b).union(
            cyl(self.P, self.s, sv.horn_r + 0.8, self.s_top - 0.5, -0.05 if mate == "horn" else 0.2))

    def face_holes(self):
        sv = self.servo
        return [(lh, wh) for lh in sv.face_holes_l for wh in (-sv.face_holes_w, sv.face_holes_w)]

    def drill(self, s0, s1, d=HOLE_M2, cb=None, cb_side=+1):
        """M2 holes through [s0, s1] (along s) at the face-hole positions; optional counterbore
        on the +s end (cb_side=+1) or the -s end (cb_side=-1)."""
        cut = None
        for lh, wh in self.face_holes():
            c = self.at(0, lh, wh)
            h = cyl(c, self.s, d / 2, s0 - 1, s1 + 1)
            if cb:
                if cb_side > 0:
                    h = h.union(cyl(c, self.s, cb[0] / 2, s1 - cb[1], s1 + 1))
                else:
                    h = h.union(cyl(c, self.s, cb[0] / 2, s0 - 1, s0 + cb[1]))
            cut = h if cut is None else cut.union(h)
        return cut

    def horn_holes(self, s0, s1):
        """Horn screws on the pitch circle + centre screw hole, through [s0, s1]."""
        sv = self.servo
        cut = cyl(self.P, self.s, 2.0, s0 - 1, s1 + 1)
        for k in range(4):
            ang = math.pi / 4 + k * math.pi / 2
            c = self.at(0, sv.horn_pcd / 2 * math.cos(ang), sv.horn_pcd / 2 * math.sin(ang))
            cut = cut.union(cyl(c, self.s, sv.horn_hole / 2, s0 - 1, s1 + 1))
        return cut

    def idler_cap(self):
        """Small plate on the back face (4x M2) with a stub on the shaft axis for a fork's idler
        ear. A filleted collar spreads the stub's bending load into the plate."""
        sv = self.servo
        s0, s1 = self.s_back - CAP_T, self.s_back
        hw = sv.face_holes_w + 3.4
        sk = rrect_sketch(-6.5, sv.face_holes_l[1] + 3.0, -hw, hw, 4.0)
        cap = prism(self.at(s0), self.l, self.w, sk, 0, CAP_T)      # normal l x w = s
        stub = cyl(self.P, self.s, STUB_R, s0 - STUB_L, s0 + 0.5)
        collar = cyl(self.P, self.s, STUB_R + 2.0, s0 - COLLAR_H, s0 + 0.5)
        collar = collar.cut(cq.Workplane(cq.Solid.makeTorus(STUB_R + 2.0, COLLAR_H,
                                                            V(*self.at(s0 - COLLAR_H)), V(*self.s))))
        cap = U(cap, stub, collar)
        win = prism(self.at(0), self.l, self.w,
                    rrect_sketch(sv.face_holes_l[0] + 4.5, sv.face_holes_l[1] - 4.5,
                                 -hw + 6.5, hw - 6.5, 2.5), -100, 100)
        return C(cap, self.drill(s0, s1, cb=CB_M2, cb_side=-1),
                 cyl(self.P, self.s, 1.25, s0 - STUB_L - 1, s0 + 4), win)


def placements(sv: Servo, side: str):
    m = 1 if side == "l" else -1

    def my(v):
        return (v[0], m * v[1], v[2])
    j = {k[2:]: v[0] for k, v in JOINTS.items() if k.startswith(side + "_")}
    return {
        "shoulder_pitch": Placed(sv, j["shoulder_pitch"], my(Y), neg(X)),
        "shoulder_roll": Placed(sv, j["shoulder_roll"], X, neg(Z)),
        "arm_yaw": Placed(sv, j["arm_yaw"], neg(Z), neg(X)),
        "elbow": Placed(sv, j["elbow"], my(Y), neg(Z)),
    }


# ------------------------------------------------------------------------------------------
# The fork: one geometry, used at the shoulder roll and at the elbow
# ------------------------------------------------------------------------------------------
FORK_D = 39.0      # bridge inner face from the joint axis (> 37.6 mm: case and cap corners)
FORK_TB = 4.0      # bridge thickness
FORK_HW = 15.0     # half-width of the ears where they meet the bridge
PLATE_REACH = 35.3  # face plates end 35.3 mm from the axis (2.6 mm past the far screws): clear of FORK_D
HORN_CORNER_R = 1.5  # fork: inside fillet where the horn ear meets the bridge (sweeps right over the face plate)
PLATE_CLIP_R = FORK_D - HORN_CORNER_R   # face plates are clipped to this radius around the joint axis, so that
                                        # fillet stays PLATE-gap (1 mm) away at every angle; far screw counterbores reach 36.4


def fork(p: Placed, down):
    """U-fork around servo p: horn ear on the horn face, idler ear on the cap stub, bridge at
    FORK_D from the axis on the `down` side. Intersection of two prisms -- the side U-profile
    (filleted inside corners) and the teardrop ear outline -- like the leg forks."""
    sA0, sA1 = 0.0, EAR
    sB1 = p.s_back - CAP_T - IDLER_GAP
    sB0 = sB1 - EAR
    D0, D1 = FORK_D, FORK_D + FORK_TB
    top = -EAR_R - 2
    u = poly_sketch([(sB0, top), (sB1, top), (sB1, D0), (sA0, D0), (sA0, top), (sA1, top),
                     (sA1, D1), (sB0, D1)], r=4.0)
    w_dir = cross(p.s, down)
    A = prism(p.P, p.s, down, u, -EAR_R - 2, EAR_R + 2)
    # faceted outline: octagonal ear (flats tangent to EAR_R) running straight into the bridge
    rv = EAR_R / math.cos(math.radians(22.5))
    oct_pts = [(rv * math.cos(math.radians(a)), rv * math.sin(math.radians(a))) for a in (67.5, 112.5, 157.5, 202.5, 247.5, 292.5)]
    cb = 4.0
    outline = [oct_pts[0], oct_pts[1], oct_pts[2], oct_pts[3], oct_pts[4], oct_pts[5],
               (D1 - cb, -FORK_HW), (D1, -FORK_HW + cb), (D1, FORK_HW - cb), (D1 - cb, FORK_HW)]
    def mk_tear(flip=False):
        if STYLE["outline"] == "oct":
            return poly_sketch([(a_, -b_ if flip else b_) for a_, b_ in outline], r=0.8)
        # smooth teardrop: ear circle + two bridge-corner circles joined by their common tangents
        # (no kink where the ear meets the flank -> clean edge rounding and clean meshes)
        R0, r1 = EAR_R, CORNER_R
        c1 = (D1 - r1, FORK_HW - r1)
        dist = math.hypot(*c1)
        base = math.atan2(c1[1], c1[0])
        phi = base + math.acos((R0 - r1) / dist)            # direction of the tangent points' normal
        n = (math.cos(phi), math.sin(phi))
        t0 = (R0 * n[0], R0 * n[1])
        t1 = (c1[0] + r1 * n[0], c1[1] + r1 * n[1])
        m1 = (c1[0] + r1 * math.cos(phi / 2), c1[1] + r1 * math.sin(phi / 2))
        e1 = (D1, c1[1])
        mir = lambda q: (q[0], -q[1])
        return (cq.Sketch()
                .arc(t0, (-R0, 0.0), mir(t0)).segment(mir(t0), mir(t1)).arc(mir(t1), mir(m1), mir(e1))
                .segment(mir(e1), e1).arc(e1, m1, t1).segment(t1, t0).assemble())
    tear = mk_tear()
    nB = cross(down, w_dir)                      # == p.s
    B = prism(p.P, down, w_dir, tear, sB0 - 1, sA1 + 1)
    f = solidify(A.intersect(B))
    # Round the ears' outer edges. A true fillet here meshes badly (it runs through the corner
    # where three rounded surfaces meet), so the rounding is built from tapered cuts: one 45 deg
    # facet for "chamfer", three facets approximating a 1.3 mm radius for "fillet".
    bevels = [(1.2, 45.0)] if STYLE["edge"] == "chamfer" else [(1.4, 25.0), (0.6, 60.0)]
    skB = mk_tear(True)
    for pl, sk in ((cq.Plane(origin=p.at(sA1), xDir=down, normal=p.s), tear),
                   (cq.Plane(origin=p.at(sB0), xDir=down, normal=neg(p.s)), skB)):
        for ch, ang in bevels:
            keep = cq.Workplane(pl).workplane(offset=-ch).placeSketch(sk).extrude(-200).union(
                cq.Workplane(pl).workplane(offset=-ch).placeSketch(sk).extrude(ch, taper=ang))
            f = solidify(f.intersect(keep))
    # the face plate under the horn ear sweeps past this corner: swap the r=4 inside fillet for a
    # small one (see PLATE_CLIP_R). The idler-side corner keeps r=4.
    r = HORN_CORNER_R
    notch = prism(p.P, p.s, down, poly_sketch([(sA0 - 4.5, D0 - 4.5), (sA0, D0 - 4.5), (sA0, D0), (sA0 - 4.5, D0)]),
                  -EAR_R - 3, EAR_R + 3)
    strip = fillet_strip(add(p.P, mul(down, D0)), neg(p.s), neg(down), w_dir, r,
                         -EAR_R - 3, EAR_R + 3)
    f = solidify(U(C(f, notch), strip.intersect(B)))
    f = C(f, p.horn_holes(sA0, sA1), cyl(p.P, p.s, STUB_R + 0.3, sB0 - 1, sB1 + 1))   # 0.3 mm running clearance
    # triangulated window: two triangles with a diagonal tie (a truss panel instead of an open slot)
    qa, qb, qc, qd = (13.0, -5.5), (D0 - 4.5, -FORK_HW + 5.0), (D0 - 4.5, FORK_HW - 5.0), (13.0, 5.5)
    def shrink(tri, k=0.80):
        cx, cy = sum(t[0] for t in tri) / 3, sum(t[1] for t in tri) / 3
        return [(cx + k * (x - cx), cy + k * (y - cy)) for x, y in tri]
    for tri in ([qa, qb, qc], [qa, qc, qd]):
        f = C(f, prism(p.P, down, w_dir, poly_sketch(shrink(tri), r=STYLE["win_r"]), -200, 200))
    assert nB == tuple(float(c) for c in p.s)
    return f


# ------------------------------------------------------------------------------------------
# Parts
# ------------------------------------------------------------------------------------------
CH_X0, CH_X1 = -30.0, 23.6     # chest back / front plate outer faces
CH_Z0, CH_Z1 = 104.0, 137.5    # chest bottom / top
T_CH = 2.5                     # chest plate thickness (strength_check: chest legs)
LEG_RIB = 6.0                  # depth of the chest legs' stiffening ribs
FEET_Y = (36.0, 48.0)          # legs land inside the shelf rim, clear of the computer standoffs
NECK = (3.3, 0.0)              # head flange centre (x, y) on the chest saddle


def feet_points():
    pts = []
    for x0 in (CH_X1 - T_CH, CH_X0):
        fxc = x0 + T_CH + 3.5 if x0 > 0 else x0 - 3.2
        for sy in (1, -1):
            pts.append((fxc, sy * (FEET_Y[0] + FEET_Y[1]) / 2))
    return pts


def part_chest_frame(sv: Servo):
    """Plate box like the torso: two cheek plates screwed to the shoulder-pitch servos' horn
    faces, front + back plates with A-legs to the shelf (4x M3 through the shelf), a bottom
    plate under the servos and a top saddle for the head. Open underneath for the electronics."""
    pl = placements(sv, "l")["shoulder_pitch"]
    pr = placements(sv, "r")["shoulder_pitch"]
    y_in = pl.P[1] - sv.horn_t                  # 62.0: servo case top face
    y_out = y_in + PLATE                        # 65.0
    parts = []
    for sgn in (1, -1):                          # cheeks, xz plane
        lo, hi = sorted((sgn * y_in, sgn * y_out))
        parts.append(prism((0, 0, 0), X, Z, rrect_sketch(CH_X0, CH_X1, CH_Z0, CH_Z1, 6.0), -hi, -lo))
    for x0 in (CH_X1 - T_CH, CH_X0):            # front / back plates with A-legs, yz plane
        pts = [(-y_out, CH_Z1), (y_out, CH_Z1), (y_out, CH_Z0 + 6), (FEET_Y[1], SHELF_TOP_Z),
               (FEET_Y[0], SHELF_TOP_Z), (26.0, CH_Z0 - 2), (-26.0, CH_Z0 - 2),
               (-FEET_Y[0], SHELF_TOP_Z), (-FEET_Y[1], SHELF_TOP_Z), (-y_out, CH_Z0 + 6)]
        parts.append(prism((x0, 0, 0), Y, Z, poly_sketch(pts, r=5.0), 0, T_CH))
        fx0, fx1 = (x0, x0 + T_CH + 7.0) if x0 > 0 else (x0 - 6.5, x0 + T_CH)
        for sy in (1, -1):
            parts.append(box((fx0, sy * FEET_Y[0], SHELF_TOP_Z), (fx1, sy * FEET_Y[1], SHELF_TOP_Z + 3.0)))
            # concave fillet between leg plate and foot tab
            if x0 > 0:
                parts.append(fillet_strip((x0 + T_CH, 0, SHELF_TOP_Z + 3.0), (1, 0, 0), (0, 0, 1), Y, 3.5,
                                          sy * FEET_Y[0] if sy > 0 else -FEET_Y[1],
                                          FEET_Y[1] if sy > 0 else -FEET_Y[0]))
            else:
                parts.append(fillet_strip((x0, 0, SHELF_TOP_Z + 3.0), (-1, 0, 0), (0, 0, 1), Y, 3.5,
                                          sy * FEET_Y[0] if sy > 0 else -FEET_Y[1],
                                          FEET_Y[1] if sy > 0 else -FEET_Y[0]))
    # Stiffening rib along each leg's outer edge, pointing into the chest. A flat 2.5 mm leg bends
    # the weak way when the robot falls forward or back; the rib makes it an L-section, which is
    # several times stiffer for ~1 g per leg (strength_check.py: chest legs).
    for x0 in (CH_X1 - T_CH, CH_X0):
        xa, xb = (x0 - LEG_RIB, x0 + 0.1) if x0 > 0 else (x0 + T_CH - 0.1, x0 + T_CH + LEG_RIB)
        for sy in (1, -1):
            band = [(sy * y_out, CH_Z0 + 6.0), (sy * FEET_Y[1], SHELF_TOP_Z), (sy * (FEET_Y[1] - 2.5), SHELF_TOP_Z),
                    (sy * (y_out - 2.5), CH_Z0 + 6.0)]
            parts.append(prism((0, 0, 0), Y, Z, poly_sketch(band), xa, xb))
    parts.append(prism((0, 0, CH_Z0), X, Y, rrect_sketch(CH_X0, CH_X1, -y_out, y_out, 5.0), 0, 2.5))
    parts.append(prism((0, 0, CH_Z1 - 4.0), X, Y, rrect_sketch(CH_X0, CH_X1, -22.0, 22.0, 5.0), 0, 4.0))
    frame = U(*parts)
    for p in (pl, pr):
        frame = C(frame, p.drill(p.s_top, p.s_top + PLATE, cb=CB_M2, cb_side=+1),
                  cyl(p.P, p.s, sv.horn_r + 0.8, -PLATE - 1, 1))
    for sgn in (1, -1):        # cheek windows behind the horn
        frame = C(frame, prism((0, 0, 0), X, Z, rrect_sketch(-27.0, -14.0, 108.0, 133.0, 4.0), -200, 200)
                  .intersect(box((-40, sgn * 55, 90), (40, sgn * 70, 150))))
    frame = C(frame, prism((0, 0, 0), Y, Z, rrect_sketch(-21.0, 21.0, CH_Z0 + 5.0, CH_Z1 - 7.0, 6.0), -40, 40))
    frame = C(frame, prism((0, 0, CH_Z0), X, Y, rrect_sketch(CH_X0 + 7, CH_X1 - 7, -50.0, 50.0, 6.0), -1, 5))
    for sy in (1, -1):         # rounded windows in the A-legs
        tri = poly_sketch([(sy * 40.5, 80.0), (sy * 45.5, 80.0), (sy * 51.0, 97.0), (sy * 34.5, 97.0)], r=2.5)
        frame = C(frame, prism((0, 0, 0), Y, Z, tri, -60, 60))
    for (fx, fy) in feet_points():
        frame = C(frame, cyl((fx, fy, SHELF_TOP_Z), Z, 1.7, -1, 6))
    for (dx, dy) in HEAD_MOUNT_HOLES:
        frame = C(frame, cyl((NECK[0] + dx, NECK[1] + dy, CH_Z1), Z, 1.25, -5, 1))   # M3 self-tap
    return C(frame, pl.keepout(mate="top"), pr.keepout(mate="top"))


HEAD_R_OUT = 36.0
HOOP_T = 2.4              # hoop wall


def head_center():
    """Face disc centre: the hoop's front face is flush with the chest front."""
    return (CH_X1 - HEAD_DEPTH / 2, NECK[1], CH_Z1 + 12.0 + HEAD_R_OUT)


HEAD_MOUNT_HOLES = [(-14.0, -9.0), (-14.0, 9.0), (7.0, -9.0), (7.0, 9.0)]   # 4x M3 into the chest saddle (rel. NECK)
HEAD_DEPTH = 14.0
HEAD_GRID = 10.0          # sensor plate hole pitch (M2.5)
PLATE_X = 9.0             # sensor plate, from the hoop's back face
VISOR_T = 1.5


def part_head():
    """Halo head: a thin hoop with a drilled sensor plate inside it; the dark face disc
    (head_visor) presses into the front. Open at the back: boards, a camera and their wires are
    all reachable.

    Interface = the neck flange (4x M3 into the chest saddle, HEAD_MOUNT_HOLES): any other head
    with this flange swaps in.
    Neck: a round column into the plate and hoop, plus a rib leaning back over the whole flange,
    so a fall on the face is carried into all four bolts instead of bending the column."""
    cx, cy, cz = head_center()
    d = HEAD_DEPTH
    x0 = cx - d / 2
    nx = NECK[0]
    hoop = cyl((x0, cy, cz), X, HEAD_R_OUT, 0, d).cut(cyl((x0, cy, cz), X, HEAD_R_OUT - HOOP_T, -1, d + 1))
    plate = cyl((x0 + PLATE_X, cy, cz), X, HEAD_R_OUT - HOOP_T + 0.5, 0, 2.0)
    col_x = x0 + 4.0
    col = cyl((col_x, cy, CH_Z1 + 3.5), Z, 9.5, 0, cz - HEAD_R_OUT + 7.0 - CH_Z1 - 3.5)
    col = col.cut(box((x0 + PLATE_X + 2.0, -50, cz - HEAD_R_OUT + 3.0), (x0 + 60, 50, cz)))    # nothing in front of the plate
    cone = cq.Workplane(cq.Solid.makeCone(14.0, 9.5, 7.0, V(col_x, cy, CH_Z1 + 3.5), V(0, 0, 1)))
    cone = cone.intersect(box((nx - 20.0, -13, CH_Z1), (CH_X1, 13, CH_Z1 + 20)))
    rib = prism((0, 0, 0), X, Z, poly_sketch([(nx - 17.0, CH_Z1 + 3.5), (x0 + 2.0, CH_Z1 + 3.5),
                                               (x0 + 2.0, cz - HEAD_R_OUT + 12.0), (x0 - 3.0, cz - HEAD_R_OUT + 12.0)], r=1.5),
                -(cy + 2.5), -(cy - 2.5))
    flange = prism((0, 0, CH_Z1), X, Y, rrect_sketch(nx - 20.0, CH_X1, cy - 13.0, cy + 13.0, 5.0), 0, 4.0)
    head = U(hoop, plate, col, cone, rib, flange)
    n = int(20 // HEAD_GRID)
    for iy in range(-n, n + 1):
        for iz in range(-n, n + 1):
            if (iy, iz) not in ((0, 1), (0, -2), (0, -1)):
                head = C(head, cyl((x0 + PLATE_X - 1, cy + iy * HEAD_GRID, cz + iz * HEAD_GRID), X, 1.35, 0, 4))
    head = C(head, box((x0 + PLATE_X - 1, cy - 6, cz + 4), (x0 + PLATE_X + 3, cy + 6, cz + 16)))     # camera lens / ribbon
    for sg in (-1, 1):                                                                             # weight + wire windows
        head = C(head, prism((0, 0, 0), Y, Z, rrect_sketch(cy + sg * 27.5 - 3.5, cy + sg * 27.5 + 3.5, cz - 16, cz + 16, 3.5),
                             x0 + PLATE_X - 1, x0 + PLATE_X + 3))
    head = C(head, cyl((col_x, cy, CH_Z1 - 1), Z, 5.0, 0, cz - HEAD_R_OUT + 3.0 - CH_Z1))        # wire duct up the neck
    head = C(head, box((col_x - 5.0, cy - 5.0, cz - HEAD_R_OUT + 1.0), (x0 + PLATE_X - 0.5, cy + 5.0, cz - HEAD_R_OUT + 7.5)))  # duct exit behind the plate
    for (dx, dy) in HEAD_MOUNT_HOLES:
        head = C(head, cyl((nx + dx, cy + dy, CH_Z1), Z, 1.7, -1, 8).union(
            cyl((nx + dx, cy + dy, CH_Z1 + 4.0), Z, 3.2, 0, 20)))       # M3 head clearance above the flange
    return head


def part_head_visor():
    """Face disc: press-fits into the hoop in front of the sensor plate (print dark, or cut from
    tinted acrylic). One camera hole; drill more as needed."""
    cx, cy, cz = head_center()
    x0 = cx - HEAD_DEPTH / 2
    v = cyl((x0 + HEAD_DEPTH - VISOR_T, cy, cz), X, HEAD_R_OUT - HOOP_T - 0.15, 0, VISOR_T)
    return C(v, cyl((x0, cy, cz + 10.0), X, 5.0, 0, 30))


def part_shoulder_bracket(sv: Servo, P):
    """Horn pad on the shoulder-pitch horn (xz plane) + face plate on the shoulder-roll servo's
    horn face (yz plane), joined at the corner with a concave fillet (like the leg's upper hip)."""
    pitch, roll = P["shoulder_pitch"], P["shoulder_roll"]
    sx, sy, sz = SHOULDER
    rx, ry, rz = ROLL
    x_in, x_out = rx - sv.horn_t, rx - sv.horn_t + PLATE        # 25.5 .. 28.5
    y0, y1 = sy, sy + EAR                                       # 66 .. 70
    pad_sk = poly_sketch([(sx, sz - EAR_R * 0.9), (x_out, rz - 27.0), (x_out, rz + 14.0),
                          (sx, sz + EAR_R * 0.9)], r=3.0)
    pad_sk = pad_sk.push([(sx, sz)]).circle(EAR_R, mode="a").clean()
    pad = prism((0, 0, 0), X, Z, pad_sk, -y1, -y0)              # normal of (X, Z) is -Y
    pz0 = rz - PLATE_REACH          # stays inside the fork bridge radius (FORK_D) at any roll angle
    pz1 = rz + 14.0
    plate_sk = poly_sketch([(y0, rz - 27.0), (ry - 13.5, pz0), (ry + 13.5, pz0), (ry + 13.5, pz1), (y0, pz1)], r=5.0)
    plate = prism((0, 0, 0), Y, Z, plate_sk, x_in, x_out).intersect(cyl(roll.P, roll.s, PLATE_CLIP_R, -20, 20))
    fil = fillet_strip((x_in, y1, 0), (-1, 0, 0), (0, 1, 0), Z, FILLET, rz - 24.0, rz + 12.0)
    part = U(pad, plate, fil)
    part = C(part, pitch.horn_holes(0, EAR),
             roll.drill(roll.s_top, roll.s_top + PLATE, cb=CB_M2, cb_side=+1),
             cyl(roll.P, roll.s, sv.horn_r + 0.8, -sv.horn_t - 1, 1),
             prism((0, 0, 0), Y, Z, rrect_sketch(ry - 6.0, ry + 6.0, rz - 29.0, rz - 15.0, 3.0), -200, 200))
    return C(part, roll.keepout(mate="top"), pitch.keepout(CLR_MOVE, mate="horn"))


def part_upper_arm(sv: Servo, P):
    """Fork around the shoulder-roll servo; its bridge is the face plate for the arm-yaw servo's
    back face (the yaw servo hangs below it)."""
    roll, yaw = P["shoulder_roll"], P["arm_yaw"]
    f = fork(roll, neg(Z))
    f = C(f, yaw.drill(yaw.s_back - FORK_TB - 0.5, yaw.s_back + 0.5, cb=CB_M2, cb_side=-1))
    lo, hi = yaw.case_bounds()
    l8, l32 = yaw.at(0, sv.face_holes_l[0])[0], yaw.at(0, sv.face_holes_l[1])[0]
    win = box((min(l8, l32) + 4.0, yaw.P[1] - 6.0, hi[2] - 1), (max(l8, l32) - 4.0, yaw.P[1] + 6.0, hi[2] + 10))
    return C(f, win, roll.keepout(CLR_MOVE, mate="horn"), yaw.keepout(mate="back"))


def part_elbow_link(sv: Servo, P):
    """Horn pad on the arm-yaw horn (xy plane, faces down) + face plate on the elbow servo's
    horn face (xz plane, outer side), joined with a concave fillet."""
    yaw, el = P["arm_yaw"], P["elbow"]
    yx, yy, yz = YAW
    ex, ey, ez = ELBOW
    z1, z0 = yz, yz - EAR
    y_in, y_out = ey - sv.horn_t, ey - sv.horn_t + PLATE
    pad_sk = poly_sketch([(yx - 14.0, yy), (yx + 14.0, yy), (yx + 14.0, y_out), (yx - 14.0, y_out)], r=3.0)
    pad_sk = pad_sk.push([(yx, yy)]).circle(EAR_R, mode="a").clean()
    pad = prism((0, 0, 0), X, Y, pad_sk, z0, z1)
    pz0 = ez - PLATE_REACH
    plate = prism((0, 0, 0), X, Z, rrect_sketch(ex - 13.5, ex + 13.5, pz0, z1, 5.0), -y_out, -y_in) \
        .intersect(cyl(el.P, el.s, PLATE_CLIP_R, -20, 20))
    fil = fillet_strip((0, y_in, z0), (0, -1, 0), (0, 0, -1), X, FILLET, ex - 13.0, ex + 13.0)
    part = U(pad, plate, fil)
    part = C(part, yaw.horn_holes(0, EAR),
             el.drill(el.s_top, el.s_top + PLATE, cb=CB_M2, cb_side=+1),
             cyl(el.P, el.s, sv.horn_r + 0.8, -sv.horn_t - 1, 1),
             prism((0, 0, 0), X, Z, rrect_sketch(ex - 6.0, ex + 6.0, ez - 26.0, ez - 15.0, 3.0), -200, 200))
    return C(part, yaw.keepout(CLR_MOVE, mate="horn"), el.keepout(mate="top"))


def rrect(w, h, r):
    return cq.Sketch().rect(w, h).vertices().fillet(r)


def finger(p0, d1, L1, d2, L2, dia):
    r = dia / 2
    v1, v2 = V(*d1).normalized(), V(*d2).normalized()
    p1 = V(*p0) + v1 * L1
    seg1 = cq.Workplane(cq.Solid.makeCylinder(r, L1 + r, V(*p0) - v1 * r, v1))
    ak = v1.cross(v2)
    ak = ak.normalized() if ak.Length > 1e-6 else V(1, 0, 0)
    knuckle = cq.Workplane(cq.Solid.makeCylinder(r, dia, p1 - ak * r, ak))
    ref = V(0, 0, 1) if abs(v2.z) < 0.9 else V(1, 0, 0)
    pl = cq.Plane(origin=p1.toTuple(), xDir=v2.toTuple(), normal=ref.cross(v2).normalized().toTuple())
    tip = (cq.Workplane(pl).moveTo(0, 0).lineTo(0, r).lineTo(L2 - 2.0, r).lineTo(L2, r - 1.6)
           .lineTo(L2, 0).close().revolve(360, (0, 0, 0), (1, 0, 0)))
    return U(seg1, knuckle, tip)


WRIST_PLATE = (16.0, 31.0, 4.0)   # wrist interface outline (x, y) and corner radius
SHELL_T = 1.5                     # forearm shell wall
WRIST_T = 5.0                     # forearm-side plate thickness (the screws thread into it)
TOOL_T = 3.0                      # tool-side plate thickness
WRIST_SCREW_X, WRIST_SCREW_Y = 4.5, 12.0   # 4x M3 at (+-4.5, +-12) from the wrist centre, screwed in from below


def wrist_frame():
    """Wrist interface (left arm): centre (x, y), z of the mating plane."""
    G = gripper_geom()
    return (ELBOW[0] - 1.0, G["yc"], G["h_z1"])


def _wrist_outline(z0, z1):
    wx, wy, _ = wrist_frame()
    a, b, r = WRIST_PLATE
    return prism((0, 0, z0), X, Y, rrect_sketch(wx - a / 2, wx + a / 2, wy - b / 2, wy + b / 2, r), 0, z1 - z0)


def part_forearm(sv: Servo, P):
    """Fork around the elbow servo + a thin hollow shell (open at the back for the wires) that
    ends in the standard wrist plate. Any tool with the matching plate bolts on with 4x M3."""
    el = P["elbow"]
    f = fork(el, neg(Z))
    ex, ey, ez = ELBOW
    wx, wy, wz = wrist_frame()
    yc = wy
    zb = ez - FORK_D - FORK_TB
    top, bot = zb + 0.6, wz + WRIST_T - 0.5
    a, b, r = WRIST_PLATE
    o1 = rrect_sketch(-9.5, 9.5, -22, 22, 6.0)      # inside the flat part of the bridge underside
    o2 = rrect_sketch(-a / 2, a / 2, -b / 2, b / 2, r).moved(cq.Location(V(wx - ex, 0, bot - top)))
    i1 = rrect_sketch(-9.5 + SHELL_T, 9.5 - SHELL_T, -22 + SHELL_T, 22 - SHELL_T, 4.6).moved(cq.Location(V(0, 0, 1.5)))
    i2 = rrect_sketch(-a / 2 + SHELL_T, a / 2 - SHELL_T, -b / 2 + SHELL_T, b / 2 - SHELL_T, r - SHELL_T + 0.3).moved(cq.Location(V(wx - ex, 0, bot - top + 0.5)))
    outer = cq.Workplane("XY", origin=(ex, yc, top)).placeSketch(o1, o2).loft()
    inner = cq.Workplane("XY", origin=(ex, yc, top)).placeSketch(i1, i2).loft()
    shell = outer.cut(inner).cut(box((ex - 30, yc - 9, bot + 3.0), (ex - 4, yc + 9, top - 2.0)))   # rear wire opening
    plate = _wrist_outline(wz, wz + WRIST_T)
    part = U(f, shell, plate)
    # open the bridge inside the shell: saves weight and lets the elbow servo's lead drop straight in
    part = C(part, prism((0, 0, zb - 1), X, Y, rrect_sketch(ex - 5.5, ex + 5.5, yc - 17.0, yc + 17.0, 3.0), 0, FORK_TB + 2))
    for sx in (-1, 1):
        for sg in (-1, 1):                                                                    # M3 self-tap pilots
            part = C(part, cyl((wx + sx * WRIST_SCREW_X, wy + sg * WRIST_SCREW_Y, wz), Z, 1.25, -1, WRIST_T + 1))
    part = C(part, prism((0, 0, wz - 1), X, Y, rrect_sketch(wx - 4, wx + 4, wy - 5, wy + 5, 2.0), 0, WRIST_T + 2))  # wire pass
    return C(part, el.keepout(CLR_MOVE, mate="horn"))


def part_claw_base():
    """Claw module: wrist plate + micro-servo housing + the fixed jaw. Bolts to the wrist plate."""
    G = gripper_geom()
    ex = ELBOW[0]
    wx, wy, wz = wrist_frame()
    yc = wy
    px, py, pz = G["pivot"]
    hx0, hx1 = ex - 9.0, G["face_x"]
    hw = GR_W / 2 + 2.0
    house = prism((0, 0, 0), Y, Z, rrect_sketch(yc - hw, yc + hw, G["h_z0"], G["h_z1"], 4.0), hx0, hx1)
    plate = _wrist_outline(wz - TOOL_T, wz)
    spine = box((hx0, yc - hw, pz - 19.0), (G["jaw_x1"], yc - 1.5, G["h_z0"] + 0.5))
    # fixed jaw: takes a fall on the hand directly (the moving jaw is protected by its servo
    # giving way), so it is 10 mm wide at the root and 7 mm thick (strength_check.py)
    fj = [(-8.4, -12.0), (-16.5, -17.0), (-19.5, -26.0), (-17.0, -36.0), (-8.0, -44.5), (-3.5, -42.5),
          (-8.5, -33.0), (-9.5, -26.0), (-8.0, -19.0), (-1.5, -19.0), (-1.5, -12.0)]
    fixed = prism((0, py, pz), Y, Z, poly_sketch(fj, r=STYLE["jaw_r"]), G["jaw_x0"] - 2.0, G["jaw_x1"])
    part = U(house, plate, spine, fixed)
    pocket = box((hx0 - 1, yc - GR_W / 2, G["p_z0"]), (hx1 + 1, yc + GR_W / 2, G["p_z1"]))
    part = C(part, pocket)
    for zt in G["tab_z"]:
        part = C(part, cyl((hx1, yc, zt), X, 0.8, -9.0, 1.0))
    for sx in (-1, 1):
        for sg in (-1, 1):                                                                    # M3 clearance
            part = C(part, cyl((wx + sx * WRIST_SCREW_X, wy + sg * WRIST_SCREW_Y, wz - TOOL_T), Z, 1.6, -1, TOOL_T + 1))
    return part


GR_L, GR_W = 23.4, 12.8      # micro-servo pocket (MG90S body 22.8 x 12.2 + clearance)
CLAW_SERVO_KG = 0.0134       # MG90S


def octo(a, b, c):
    return cq.Sketch().polygon([(a - c, b), (a, b - c), (a, -b + c), (a - c, -b), (-a + c, -b), (-a, -b + c),
                                (-a, b - c), (-a + c, b), (a - c, b)])


def octo_at(cx, cy, a, b, c):
    return cq.Sketch().polygon([(cx + x, cy + y) for x, y in
                                [(a - c, b), (a, b - c), (a, -b + c), (a - c, -b), (-a + c, -b), (-a, -b + c),
                                 (-a, b - c), (-a + c, b), (a - c, b)]])


def gripper_geom():
    """Wrist servo layout (left arm). Shaft along +X, body length along Z, shaft at the lower end."""
    sv = STS3215
    ex, ey, ez = ELBOW
    P = placements(sv, "l")["elbow"]
    yB = P.at(P.s_back - CAP_T - IDLER_GAP - EAR)[1]
    yc = (ey + EAR + yB) / 2
    wrist_z = ez - FOREARM_TO_TIP + 57.0
    p_z1 = wrist_z - 4.0                 # pocket top
    p_z0 = p_z1 - GR_L
    face_x = ex + 7.0                    # housing front face = servo tab seat
    return {"yc": yc, "p_z0": p_z0, "p_z1": p_z1, "h_z0": p_z0 - 6.5, "h_z1": p_z1 + 6.5, "face_x": face_x,
            "tab_z": ((p_z0 + p_z1) / 2 - 13.9, (p_z0 + p_z1) / 2 + 13.9),
            "pivot": (face_x, yc, p_z0 + 6.2), "jaw_x0": face_x + 11.0, "jaw_x1": face_x + 16.0}


def part_claw_jaw():
    """Moving jaw: 5 mm plate on the micro servo's horn (closed pose)."""
    G = gripper_geom()
    px, py, pz = G["pivot"]
    mj = [(-5.0, 6.5), (6.0, 6.5), (14.0, -6.0), (17.5, -20.0), (15.0, -34.0), (7.0, -44.0), (3.5, -42.5),
          (9.5, -33.0), (11.5, -20.0), (8.0, -9.0), (1.0, -7.5), (-5.0, -7.0)]
    jaw = prism((0, py, pz), Y, Z, poly_sketch(mj, r=STYLE["jaw_r"]), G["jaw_x0"], G["jaw_x1"])
    cut = cyl((G["jaw_x0"], py, pz), X, 1.3, -1, 6).union(cyl((G["jaw_x0"], py, pz), X, 3.6, -1, 2.0))
    win = prism((0, py, pz), Y, Z, poly_sketch([(10.5, -12.0), (13.5, -20.0), (12.0, -28.0), (14.5, -20.0)], r=0.3), -200, 200)
    return C(jaw, cut)


def gripper_servo():
    G = gripper_geom()
    yc = G["yc"]
    body = box((G["face_x"] - 16.0, yc - 6.1, G["p_z0"] + 0.3), (G["face_x"] + 6.8, yc + 6.1, G["p_z1"] - 0.3))
    tabs = box((G["face_x"], yc - 6.1, G["tab_z"][0] - 2.4), (G["face_x"] + 2.5, yc + 6.1, G["tab_z"][1] + 2.4))
    px, py, pz = G["pivot"]
    return body.union(tabs).union(cyl((px, py, pz), X, 5.8, 6.8, 9.0)).union(cyl((px, py, pz), X, 2.4, 9.0, 11.0))


def mirror_y(wp):
    return wp.mirror("XZ")


def build(variant: str = "sts3215"):
    sv = SERVOS[variant]
    P = placements(sv, "l")
    parts = {
        "chest_frame": part_chest_frame(sv),
        "head": part_head(),
        "head_visor": part_head_visor(),
        "l_shoulder_bracket": part_shoulder_bracket(sv, P),
        "l_roll_idler_cap": P["shoulder_roll"].idler_cap(),
        "l_upper_arm": part_upper_arm(sv, P),
        "l_elbow_link": part_elbow_link(sv, P),
        "l_elbow_idler_cap": P["elbow"].idler_cap(),
        "l_forearm": part_forearm(sv, P),
        "l_claw_base": part_claw_base(),
        "l_claw_jaw": part_claw_jaw(),
    }
    for k in ["shoulder_bracket", "roll_idler_cap", "upper_arm", "elbow_link", "elbow_idler_cap", "forearm",
              "claw_base", "claw_jaw"]:
        parts["r_" + k] = mirror_y(parts["l_" + k])
    servos = {f"{side}_{jn}": p for side in "lr" for jn, p in placements(sv, side).items()}
    return sv, parts, servos


def drill_template():
    t = box((-38, -48.5, SHELF_TOP_Z), (42, 48.5, SHELF_TOP_Z + 1.2))
    t = t.cut(box((-30, -34, SHELF_TOP_Z - 1), (34, 34, SHELF_TOP_Z + 3)))
    for (fx, fy) in feet_points():
        t = t.cut(cyl((fx, fy, SHELF_TOP_Z), Z, 1.7, -1, 3))
    return t


def export_stl(wp, path):
    import trimesh
    for tol, ang in ((0.05, 0.2), (0.02, 0.1), (0.1, 0.5), (0.03, 0.15), (0.08, 0.3), (0.04, 0.25)):
        cq.exporters.export(wp, str(path), tolerance=tol, angularTolerance=ang)
        if trimesh.load(str(path)).is_watertight:
            return True
    return False


def export(variant: str = "sts3215"):
    sv, parts, servos = build(variant)
    parts["drill_template"] = drill_template()
    d = OUT / variant
    for sub in ("step", "stl", "servo_standins", "extras"):
        (d / sub).mkdir(parents=True, exist_ok=True)
        for f in (d / sub).glob("*"):
            f.unlink()
    info = {"variant": variant, "servo": sv.__dict__, "parts": {}, "servos": {}}
    for name, wp in parts.items():
        s = wp.val()
        cq.exporters.export(wp, str(d / "step" / f"{name}.step"))
        wt = export_stl(wp, d / "stl" / f"{name}.stl")
        bb = s.BoundingBox()
        info["parts"][name] = {"volume_cm3": round(s.Volume() / 1000, 3), "solids": len(s.Solids()),
                               "valid": s.isValid(), "watertight_stl": wt,
                               "bbox_mm": [round(v, 2) for v in (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)]}
    for name, p in servos.items():
        cq.exporters.export(p.solid(), str(d / "servo_standins" / f"{name}.stl"), tolerance=0.1)
        info["servos"][name] = {"case_bounds_mm": [[round(v, 3) for v in b] for b in p.case_bounds()]}
    # bought parts that are not STS3215s: the claw micro servos (stand-in solid + mass)
    info["extras"] = {}
    g = gripper_servo()
    for side, wp in (("l", g), ("r", mirror_y(g))):
        name = f"{side}_claw_servo"
        cq.exporters.export(wp, str(d / "extras" / f"{name}.stl"), tolerance=0.1)
        bb = wp.val().BoundingBox()
        info["extras"][name] = {"mass_kg": CLAW_SERVO_KG, "link": f"{side}_forearm_link",
                                "bounds_mm": [[round(bb.xmin, 2), round(bb.ymin, 2), round(bb.zmin, 2)],
                                              [round(bb.xmax, 2), round(bb.ymax, 2), round(bb.zmax, 2)]]}
    (d / "parts.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    info = export("sts3215")
    for k, v in info["parts"].items():
        print(f"{k:22s} {v['volume_cm3']:7.2f} cm3  solids={v['solids']} valid={v['valid']} watertight={v['watertight_stl']}")
