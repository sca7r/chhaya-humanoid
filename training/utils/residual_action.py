"""Residual action map helpers (action_map: residual). numpy + math only: no Isaac, no torch.

    target = clamp(home + scale_j * clamp(a, -1, 1), joint_min, joint_max)
    a      = clamp((target - home) / scale_j, -1, 1)

Asymmetric scales (OFF by default): scale_j is s_pos_j for a >= 0 and s_neg_j for a < 0 (a = 0 is still the home pose),
    target = clamp(home + s_pos_j * a, ...) for a >= 0,  clamp(home + s_neg_j * a, ...) for a < 0
    a      = clamp((target - home) / s_pos_j, -1, 1) if target >= home else clamp((target - home) / s_neg_j, -1, 1)
"pos" is the joint's positive direction (a > 0 = joint angle above home), "neg" the other one. s_pos == s_neg is the
symmetric map above, bit for bit (every consumer keeps its old expression when there is no neg array).

`home` is the crouched, flat-footed pose that a = 0 means; it is also the articulation's
default_joint_pos, so every "relative to default" quantity (joint_pos_rel obs, joint_deviation_*,
dof_pos_l2, knee bend) is measured from it. Both are configured BY JOINT TYPE in config.yaml
(env_config.action_home_deg / action_scale_rad, same value for the left and right joint) and
expanded here to the 12 policy slots. action_scale_rad[type] is a number (symmetric) or {pos: x, neg: y}.

Policy joint order (Isaac, MuJoCo and the exported bundle all agree): L,R interleaved per type,
    0 l_hip_yaw  1 r_hip_yaw  2 l_hip_roll  3 r_hip_roll  4 l_hip_pitch  5 r_hip_pitch
    6 l_knee     7 r_knee     8 l_ankle_roll 9 r_ankle_roll 10 l_ankle_pitch 11 r_ankle_pitch
"""
from __future__ import annotations

import math

import numpy as np

JOINT_TYPES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll", "ankle_pitch")

# Spawn base height for the residual home: the V2 default of 0.29 m minus the 3.9 mm the
# crouch lowers the base (home hip -10 / knee -26 / ankle -16 deg).
RESIDUAL_SPAWN_HEIGHT_M = 0.286


def expand_by_type(values: dict, name: str) -> list[float]:
    """{type: value} -> 12 floats in policy order (each type repeated for l, r)."""
    values = dict(values or {})
    unknown = set(values) - set(JOINT_TYPES)
    missing = set(JOINT_TYPES) - set(values)
    if unknown or missing:
        raise ValueError(f"{name}: need exactly the keys {list(JOINT_TYPES)}; "
                         f"missing {sorted(missing)}, unknown {sorted(unknown)}")
    return [float(values[t]) for t in JOINT_TYPES for _ in range(2)]


def home_rad(action_home_deg: dict) -> np.ndarray:
    return np.radians(np.asarray(expand_by_type(action_home_deg, "action_home_deg"), dtype=np.float64))


SCALE_SIDES = ("pos", "neg")


def split_scale(value, name: str = "action_scale_rad") -> tuple[float, float]:
    """One joint type's scale -> (s_pos, s_neg): a number is symmetric, {pos: x, neg: y} is not."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        if set(value) != set(SCALE_SIDES):
            raise ValueError(f"{name}: a per-side scale needs exactly the keys {list(SCALE_SIDES)}, got {sorted(value)}")
        return float(value["pos"]), float(value["neg"])
    return float(value), float(value)


def plain_scale_cfg(action_scale_rad: dict) -> dict:
    """{type: float | {pos: float, neg: float}}: a type whose two sides are equal collapses to its number."""
    out = {}
    for t, v in dict(action_scale_rad).items():
        p, n = split_scale(v, f"action_scale_rad.{t}")
        out[t] = p if p == n else {"pos": p, "neg": n}
    return out


def scale_pair_rad(action_scale_rad: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """{type: number | {pos, neg}} -> (pos [12], neg [12] or None). neg is None when every type is symmetric (pos == neg),
    so callers keep their original symmetric expression; both arrays are > 0 and finite."""
    values = dict(action_scale_rad or {})
    pairs = {t: split_scale(v, f"action_scale_rad.{t}") for t, v in values.items()}
    pos = np.asarray(expand_by_type({t: p[0] for t, p in pairs.items()}, "action_scale_rad"), dtype=np.float64)
    neg = np.asarray(expand_by_type({t: p[1] for t, p in pairs.items()}, "action_scale_rad"), dtype=np.float64)
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(neg)) and np.all(pos > 0.0) and np.all(neg > 0.0)):
        raise ValueError(f"action_scale_rad must be finite and > 0 on both sides for every joint type, got {action_scale_rad}")
    return pos, (None if np.array_equal(pos, neg) else neg)


def is_asymmetric_scale(action_scale_rad) -> bool:
    return scale_pair_rad(action_scale_rad)[1] is not None


def scale_rad(action_scale_rad: dict) -> np.ndarray:
    """Symmetric scales [12]. Raises for a map with pos != neg (use scale_pair_rad)."""
    pos, neg = scale_pair_rad(action_scale_rad)
    if neg is not None:
        raise ValueError("action_scale_rad is asymmetric (pos != neg): use scale_pair_rad")
    return pos


# ----------------------------------------------------------------------------------------------- joint-limit override
# env_config.joint_limits_deg: {type: [min_deg, max_deg]} (l and r share a row, as for the home). Absent = the robot's
# _JOINT_LIMITS_DEG (envs/assets/robotV2/biped_robot.py), the walking box. The get-up config sets it to the URDF limits
# x 0.9. It is the action clamp, the joint_pos_limits penalty box and the bundle's joint_limits_min/max, so every
# consumer (Isaac env, MuJoCo env, export, residual resolver) must read it from the run config, never the constant.

BOX_SCALE = "box"


def limits_rows_deg(joint_limits_deg) -> np.ndarray:
    """{type: [min, max]} [deg] -> [12, 2] in policy order. Raises on missing / unknown types or min >= max."""
    if hasattr(joint_limits_deg, "model_dump"):
        joint_limits_deg = joint_limits_deg.model_dump()
    d = dict(joint_limits_deg or {})
    unknown, missing = set(d) - set(JOINT_TYPES), set(JOINT_TYPES) - set(d)
    if unknown or missing:
        raise ValueError(f"joint_limits_deg: need exactly the keys {list(JOINT_TYPES)}; "
                         f"missing {sorted(missing)}, unknown {sorted(unknown)}")
    rows = []
    for t in JOINT_TYPES:
        lo, hi = (float(v) for v in d[t])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"joint_limits_deg.{t}: [{lo}, {hi}] must be finite with min < max")
        rows += [[lo, hi], [lo, hi]]
    return np.asarray(rows, dtype=np.float64)


def is_box_scale(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == BOX_SCALE


def box_scale_cfg(action_home_deg: dict, limits_deg_12) -> dict:
    """action_scale_rad: box -> {type: {pos: max - home, neg: home - min}} [rad], so a = +-1 reaches the limit box edges
    exactly (a = 0 stays the home). Rounded DOWN to 1e-4 rad so the clamp, not float noise, decides the edge."""
    home = home_rad(action_home_deg)
    lim = np.radians(np.asarray(limits_deg_12, dtype=np.float64))
    out = {}
    for i, t in enumerate(JOINT_TYPES):
        pos, neg = lim[2 * i, 1] - home[2 * i], home[2 * i] - lim[2 * i, 0]
        if pos <= 0.0 or neg <= 0.0:
            raise ValueError(f"action_scale_rad: box needs the home strictly inside the limits ({t})")
        out[t] = {"pos": math.floor(pos * 1e4) / 1e4, "neg": math.floor(neg * 1e4) / 1e4}
    return out


def check_residual_map(home, scale, lo, hi, mirror_perm=None, mirror_signs=None, scale_neg=None) -> None:
    """Raise unless the map is usable: home strictly inside the limits, scale > 0 (both sides when scale_neg is given), and
    home invariant under the left/right mirror (home[l] == home[r] for the all-+1 V2 signs)."""
    if scale_neg is not None:
        check_residual_map(home, scale_neg, lo, hi, mirror_perm, mirror_signs)
    home, scale = np.asarray(home, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if home.shape != (12,) or scale.shape != (12,):
        raise ValueError(f"home/scale must have 12 entries, got {home.shape}/{scale.shape}")
    if np.any(scale <= 0.0):
        raise ValueError("action scale must be > 0")
    if np.any(home <= lo) or np.any(home >= hi):
        raise ValueError("residual home must be strictly inside the joint limits")
    perm = [i ^ 1 for i in range(12)] if mirror_perm is None else list(mirror_perm)  # (0,1),(2,3).. L<->R
    signs = np.ones(12) if mirror_signs is None else np.asarray(mirror_signs, dtype=np.float64)
    if not np.allclose(home[perm] * signs, home, atol=1e-9) or not np.allclose(scale[perm], scale, atol=1e-9):
        raise ValueError("residual home/scale must be left/right symmetric (mirror augmentation "
                         "and the exported mirror table assume it)")


def actions_to_targets_np(a, home, scale, lo, hi, scale_neg=None):
    """scale = s_pos [12]; scale_neg = s_neg [12] for the asymmetric map (None = symmetric, the original expression)."""
    a = np.clip(a, -1.0, 1.0)
    if scale_neg is not None:
        scale = np.where(a >= 0.0, scale, scale_neg)
    return np.clip(home + scale * a, lo, hi)


def targets_to_actions_np(t, home, scale, scale_neg=None):
    """Inverse of actions_to_targets_np (clipped to [-1, 1])."""
    d = np.asarray(t) - home
    if scale_neg is not None:
        scale = np.where(d >= 0.0, scale, scale_neg)
    return np.clip(d / scale, -1.0, 1.0)
