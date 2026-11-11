"""Fall bank: settled fallen states the get-up policy starts from. numpy only (no Isaac, no torch, no mujoco).

Written by src/isaaclab/make_fall_bank.py, read by envs/isaaclab/getup_env.py (training starts) and
src/mujoco/getup_gate.py (held-out gate starts). One state per row:

    root_z           [M]      base_link (Isaac root) height above the floor [m]
    root_quat        [M, 4]   base orientation (w, x, y, z) with the YAW REMOVED (the env draws a random heading)
    joint_pos        [M, 12]  joint angles [rad], policy order (L,R interleaved per type, utils.residual_action)
    joint_vel        [M, 12]  [rad/s]
    root_lin_vel_b   [M, 3]   base linear velocity in the base frame [m/s]
    root_ang_vel_b   [M, 3]   base angular velocity in the base frame [rad/s]
    category         [M]      index into CATEGORIES, from the body-frame gravity at capture
    source           [M]      index into SOURCES (how the fall was produced)
    split            [M]      0 = train, 1 = gate (held out: the MuJoCo gate never sees a training start)
    mirrored         [M]      True for the exact mirror copy of another row
    meta_json        []       json: joint_names, categories, sources, h_floor_m, settle criterion, generator settings

The states are what the Jetson sees when it re-arms: the robot fell with torque OFF, lay still until the settle
criterion held, and the policy starts from there (see the get-up bundle's start_protocol).

Mirror (sagittal plane, the env's y -> -y): a pure joint permutation for V2 (mirror signs are all +1, see
biped_env_cfg.BipedRobotV2EnvCfg), quaternion (w, x, y, z) -> (w, -x, y, -z), body-frame vectors: linear (x, -y, z),
angular (-x, y, -z), category left <-> right.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

FORMAT_VERSION = 1

# Dominant axis of the gravity vector in the base frame (body x forward, y left, z up). Prone = lying face down
# (body x points down, so gravity reads +x), supine = on the back (-x). left_side = lying on the left side (body y
# points down, gravity +y). upright = the base is within ~45 deg of vertical (sitting, kneeling, crouched or standing).
CATEGORIES = ("upright", "prone", "supine", "left_side", "right_side", "inverted")
SOURCES = ("drop", "push", "hold")
_MIRROR_CATEGORY = {"left_side": "right_side", "right_side": "left_side"}

# V2 policy order: mirror is the L <-> R swap of every joint type, signs +1 (biped_env_cfg.BipedRobotV2EnvCfg).
MIRROR_JOINT_PERM = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
MIRROR_JOINT_SIGNS = (1.0,) * 12

ARRAY_KEYS = ("root_z", "root_quat", "joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b",
              "category", "source", "split", "mirrored")


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """[M, 3] gravity direction (0, 0, -1) expressed in the base frame, R^T (0, 0, -1), for quats (w, x, y, z)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # third row of R (world z in body coordinates is R^T e_z = (R[2,0], R[2,1], R[2,2])), negated
    return -np.stack([2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)], axis=-1)


def categorize(grav_b: np.ndarray) -> np.ndarray:
    """[M] category index from the body-frame gravity direction (see CATEGORIES)."""
    g = np.asarray(grav_b, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(g), dtype=np.int8)
    up = g[:, 2] < -math.sqrt(0.5)          # base within 45 deg of upright
    inv = g[:, 2] > math.sqrt(0.5)
    ax = np.abs(g[:, 0]) >= np.abs(g[:, 1])
    out[:] = np.where(ax, np.where(g[:, 0] > 0, CATEGORIES.index("prone"), CATEGORIES.index("supine")),
                      np.where(g[:, 1] > 0, CATEGORIES.index("left_side"), CATEGORIES.index("right_side")))
    out[up] = CATEGORIES.index("upright")
    out[inv] = CATEGORIES.index("inverted")
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def remove_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rz(-yaw) * q with yaw the ZYX heading: the same tilt, heading 0. Canonical sign (w >= 0)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    half = -0.5 * yaw_of(q)
    rz = np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1)
    out = _quat_mul(rz, q)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(out[:, :1] < 0.0, -out, out)


def mirror_states(states: dict) -> dict:
    """Exact sagittal mirror of every row (see the module docstring). Arrays are copied, not aliased."""
    perm, signs = np.asarray(MIRROR_JOINT_PERM), np.asarray(MIRROR_JOINT_SIGNS)
    q = np.asarray(states["root_quat"], dtype=np.float64)
    cat = np.asarray(states["category"]).astype(np.int8).copy()
    swap = {CATEGORIES.index(a): CATEGORIES.index(b) for a, b in _MIRROR_CATEGORY.items()}
    cat_m = cat.copy()
    for a, b in swap.items():
        cat_m[cat == a] = b
    out = {
        "root_z": np.asarray(states["root_z"], dtype=np.float64).copy(),
        "root_quat": np.stack([q[:, 0], -q[:, 1], q[:, 2], -q[:, 3]], axis=-1),
        "joint_pos": np.asarray(states["joint_pos"], dtype=np.float64)[:, perm] * signs,
        "joint_vel": np.asarray(states["joint_vel"], dtype=np.float64)[:, perm] * signs,
        "root_lin_vel_b": np.asarray(states["root_lin_vel_b"], dtype=np.float64) * np.array([1.0, -1.0, 1.0]),
        "root_ang_vel_b": np.asarray(states["root_ang_vel_b"], dtype=np.float64) * np.array([-1.0, 1.0, -1.0]),
        "category": cat_m,
        "source": np.asarray(states["source"]).astype(np.int8).copy(),
    }
    for k in ("split", "mirrored"):
        if k in states:
            out[k] = np.asarray(states[k]).copy()
    return out


@dataclass
class FallBank:
    """In-memory bank. `arrays` holds the ARRAY_KEYS columns (float64 / int8 / bool), `meta` the json metadata."""
    arrays: dict
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.arrays["root_z"]))

    @property
    def h_floor_m(self) -> float:
        return float(self.meta["h_floor_m"])

    def select(self, split: str | None = None, category: str | None = None) -> np.ndarray:
        """Row indices of a split ("train" | "gate" | None = both) and optionally one category name."""
        m = np.ones(len(self), dtype=bool)
        if split is not None:
            if split not in ("train", "gate"):
                raise ValueError(f"split {split!r}: expected train | gate")
            m &= self.arrays["split"] == (0 if split == "train" else 1)
        if category is not None:
            if category not in CATEGORIES:
                raise ValueError(f"category {category!r}: expected one of {CATEGORIES}")
            m &= self.arrays["category"] == CATEGORIES.index(category)
        return np.flatnonzero(m)

    def histogram(self, idx: np.ndarray | None = None) -> dict:
        cat = self.arrays["category"] if idx is None else self.arrays["category"][idx]
        return {c: int((cat == i).sum()) for i, c in enumerate(CATEGORIES)}

    def row(self, i: int) -> dict:
        return {k: self.arrays[k][i] for k in ARRAY_KEYS}


def build_bank(states: dict, meta: dict, *, gate_frac: float = 0.1, seed: int = 0) -> FallBank:
    """Captured states (no mirror) -> a bank with an exact mirror of every row and a train / gate split.

    The split is drawn per ORIGINAL state and its mirror copy follows it, so a gate start is never the mirror of a
    training start. h_floor_m (the 'lying' base height of the height reward) = median root_z of the prone / supine /
    side states."""
    n = len(states["root_z"])
    if n == 0:
        raise ValueError("no captured states")
    rng = np.random.default_rng(seed)
    split = (rng.random(n) < gate_frac).astype(np.int8)
    base = {k: np.asarray(states[k]) for k in ("root_z", "root_quat", "joint_pos", "joint_vel",
                                                 "root_lin_vel_b", "root_ang_vel_b", "category", "source")}
    base["split"], base["mirrored"] = split, np.zeros(n, dtype=bool)
    mir = mirror_states(base)
    mir["mirrored"] = np.ones(n, dtype=bool)
    arrays = {k: np.concatenate([np.asarray(base[k]), np.asarray(mir[k])]) for k in ARRAY_KEYS}
    _cast(arrays)
    lying = np.isin(arrays["category"], [CATEGORIES.index(c) for c in ("prone", "supine", "left_side", "right_side")])
    z = arrays["root_z"][lying] if lying.any() else arrays["root_z"]
    meta = {**meta, "format_version": FORMAT_VERSION, "categories": list(CATEGORIES), "sources": list(SOURCES),
            "h_floor_m": float(np.median(z)), "n_states": int(len(arrays["root_z"])), "gate_frac": float(gate_frac)}
    bank = FallBank(arrays, meta)
    validate(bank)
    return bank


def _cast(arrays: dict) -> None:
    for k in ("root_z", "root_quat", "joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b"):
        arrays[k] = np.asarray(arrays[k], dtype=np.float64)
    for k in ("category", "source", "split"):
        arrays[k] = np.asarray(arrays[k]).astype(np.int8)
    arrays["mirrored"] = np.asarray(arrays["mirrored"]).astype(bool)


def validate(bank: FallBank) -> None:
    """Raise on a malformed bank: shapes, finiteness, unit quaternions, known categories, positive heights."""
    a, n = bank.arrays, len(bank)
    want = {"root_z": (n,), "root_quat": (n, 4), "joint_pos": (n, 12), "joint_vel": (n, 12),
            "root_lin_vel_b": (n, 3), "root_ang_vel_b": (n, 3), "category": (n,), "source": (n,),
            "split": (n,), "mirrored": (n,)}
    for k, shape in want.items():
        if k not in a:
            raise ValueError(f"fall bank: missing array {k!r}")
        if tuple(a[k].shape) != shape:
            raise ValueError(f"fall bank: {k} has shape {tuple(a[k].shape)}, expected {shape}")
    for k in ("root_z", "root_quat", "joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b"):
        if not np.all(np.isfinite(a[k])):
            raise ValueError(f"fall bank: non-finite values in {k}")
    if np.max(np.abs(np.linalg.norm(a["root_quat"], axis=-1) - 1.0)) > 1e-6:
        raise ValueError("fall bank: root_quat rows must be unit quaternions")
    if np.any(a["root_z"] <= 0.0):
        raise ValueError("fall bank: root_z must be > 0 (above the floor)")
    if np.any((a["category"] < 0) | (a["category"] >= len(CATEGORIES))):
        raise ValueError("fall bank: unknown category index")
    names = bank.meta.get("joint_names")
    if names is not None and len(names) != 12:
        raise ValueError("fall bank: meta.joint_names must list the 12 policy joints")


def save_bank(bank: FallBank, path: str) -> None:
    validate(bank)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, meta_json=np.array(json.dumps(bank.meta, sort_keys=True)),
                        **{k: bank.arrays[k] for k in ARRAY_KEYS})


def load_bank(path: str, root_dir: str | None = None) -> FallBank:
    """Load and validate. A relative path is resolved against root_dir (the BipedRobot repo) when given."""
    full = path if os.path.isabs(path) or root_dir is None else os.path.join(root_dir, path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"fall bank {full} not found. Generate it first (Isaac Lab python, from BipedRobot/):\n"
            f"  python src/isaaclab/make_fall_bank.py --config_path config/config_getup.yaml --headless")
    with np.load(full, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        arrays = {k: np.asarray(z[k]) for k in ARRAY_KEYS}
    if int(meta.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"fall bank {full}: format_version {meta.get('format_version')} != {FORMAT_VERSION}")
    _cast(arrays)
    bank = FallBank(arrays, meta)
    validate(bank)
    return bank
