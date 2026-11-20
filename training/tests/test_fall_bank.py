"""Fall bank (utils/fall_bank.py): gravity / categories, the exact sagittal mirror, yaw removal, the train / gate split,
save / load. numpy only (no Isaac). Run from training/:  python -m pytest tests/test_fall_bank.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import fall_bank as FB  # noqa: E402


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def _rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q = FB.remove_yaw(q)
    return {"root_z": rng.uniform(0.03, 0.25, n), "root_quat": q, "joint_pos": rng.uniform(-1, 1, (n, 12)),
            "joint_vel": rng.normal(0, 0.1, (n, 12)), "root_lin_vel_b": rng.normal(0, 0.02, (n, 3)),
            "root_ang_vel_b": rng.normal(0, 0.05, (n, 3)),
            "category": FB.categorize(FB.projected_gravity(q)), "source": rng.integers(0, 3, n).astype(np.int8)}


def test_projected_gravity_matches_rotation_matrix():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((50, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    want = np.stack([_rotmat(qi).T @ np.array([0.0, 0.0, -1.0]) for qi in q])
    assert np.allclose(FB.projected_gravity(q), want, atol=1e-12)


@pytest.mark.parametrize("axis,angle,cat", [
    ((1, 0, 0), 0.0, "upright"),
    ((0, 1, 0), math.pi / 2, "prone"),        # pitch +90 = nose down = face down
    ((0, 1, 0), -math.pi / 2, "supine"),
    ((1, 0, 0), math.pi / 2, "right_side"),   # roll +90: body y (left) points up, the right side is down
    ((1, 0, 0), -math.pi / 2, "left_side"),
    ((1, 0, 0), math.pi, "inverted"),
    ((0, 1, 0), math.radians(40), "upright"), # within 45 deg of vertical: sitting / kneeling / crouched
])
def test_categories(axis, angle, cat):
    q = _quat_axis_angle(axis, angle)[None]
    assert FB.CATEGORIES[int(FB.categorize(FB.projected_gravity(q))[0])] == cat


def test_mirror_is_an_involution_and_mirrors_gravity():
    s = _random_states(200)
    m = FB.mirror_states(s)
    mm = FB.mirror_states(m)
    for k in ("root_z", "root_quat", "joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b", "category"):
        assert np.allclose(mm[k], s[k]), k
    g, gm = FB.projected_gravity(s["root_quat"]), FB.projected_gravity(m["root_quat"])
    assert np.allclose(gm, g * np.array([1.0, -1.0, 1.0]), atol=1e-12)      # the policy obs mirror: gravity y flips
    assert np.array_equal(FB.categorize(gm), m["category"])                 # left <-> right, the rest unchanged
    assert np.allclose(m["joint_pos"], s["joint_pos"][:, list(FB.MIRROR_JOINT_PERM)])
    # body-frame rates mirror like the gyro slots of the obs mirror (x, z flip) and the critic's lin vel (y flips)
    assert np.allclose(m["root_ang_vel_b"], s["root_ang_vel_b"] * [-1, 1, -1])
    assert np.allclose(m["root_lin_vel_b"], s["root_lin_vel_b"] * [1, -1, 1])


def test_mirror_quaternion_is_the_reflected_rotation():
    """S R S with S = diag(1, -1, 1) (the world sagittal reflection) is the rotation of the mirrored quaternion."""
    rng = np.random.default_rng(2)
    S = np.diag([1.0, -1.0, 1.0])
    for _ in range(20):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        qm = FB.mirror_states({**_random_states(1), "root_quat": q[None]})["root_quat"][0]
        assert np.allclose(_rotmat(qm), S @ _rotmat(q) @ S, atol=1e-12)


def test_remove_yaw_keeps_tilt():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((100, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    r = FB.remove_yaw(q)
    assert np.allclose(FB.yaw_of(r), 0.0, atol=1e-9)
    assert np.allclose(FB.projected_gravity(r), FB.projected_gravity(q), atol=1e-9)
    assert np.all(r[:, 0] >= 0.0)


def test_build_bank_pairs_mirrors_and_split():
    s = _random_states(300)
    bank = FB.build_bank(s, {"joint_names": [f"j{i}" for i in range(12)]}, gate_frac=0.2, seed=0)
    a, n = bank.arrays, 300
    assert len(bank) == 2 * n
    assert not a["mirrored"][:n].any() and a["mirrored"][n:].all()
    assert np.array_equal(a["split"][:n], a["split"][n:])               # a gate start is never a training mirror
    m = FB.mirror_states({k: a[k][:n] for k in FB.ARRAY_KEYS})
    assert np.allclose(m["joint_pos"], a["joint_pos"][n:]) and np.allclose(m["root_quat"], a["root_quat"][n:])
    assert 0.1 < (a["split"] == 1).mean() < 0.3
    hist = bank.histogram()
    assert hist["left_side"] == hist["right_side"]                       # mirror-complete
    lying = np.isin(a["category"], [FB.CATEGORIES.index(c) for c in ("prone", "supine", "left_side", "right_side")])
    assert bank.h_floor_m == pytest.approx(float(np.median(a["root_z"][lying])))
    assert len(bank.select("train")) + len(bank.select("gate")) == len(bank)
    assert all(a["category"][i] == FB.CATEGORIES.index("supine") for i in bank.select(category="supine"))


def test_save_load_roundtrip(tmp_path):
    bank = FB.build_bank(_random_states(50), {"joint_names": [f"j{i}" for i in range(12)]}, seed=1)
    path = str(tmp_path / "bank.npz")
    FB.save_bank(bank, path)
    back = FB.load_bank(path)
    for k in FB.ARRAY_KEYS:
        assert np.array_equal(back.arrays[k], bank.arrays[k]), k
    assert back.meta["h_floor_m"] == bank.meta["h_floor_m"]


def test_validate_rejects_bad_rows():
    bank = FB.build_bank(_random_states(20), {}, seed=0)
    bank.arrays["root_quat"][0] *= 2.0
    with pytest.raises(ValueError, match="unit quaternions"):
        FB.validate(bank)


def test_missing_bank_names_the_generator(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_fall_bank"):
        FB.load_bank(str(tmp_path / "nope.npz"))
