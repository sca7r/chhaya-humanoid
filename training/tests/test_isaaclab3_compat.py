"""Isaac Lab 3.0 boundary: quaternion order and the (w, x, y, z) helpers (envs/isaaclab/compat.py, quat.py).

No simulator needed. The reference for the quaternion maths is MuJoCo (also w, x, y, z).
"""
import math
import os
import sys

import mujoco
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from envs.isaaclab import quat as Q  # noqa: E402
from envs.isaaclab.compat import TorchData, wxyz_to_xyzw, xyzw_to_wxyz  # noqa: E402


def _rand_quat(n):
    q = torch.randn(n, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def test_order_round_trip():
    q = _rand_quat(8)
    assert torch.equal(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
    assert torch.equal(wxyz_to_xyzw(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([2.0, 3.0, 4.0, 1.0]))


def test_rotation_and_product_match_mujoco():
    torch.manual_seed(0)
    q, p, v = _rand_quat(16), _rand_quat(16), torch.randn(16, 3, dtype=torch.float64)
    for i in range(16):
        out = np.zeros(3); mujoco.mju_rotVecQuat(out, v[i].numpy(), q[i].numpy())
        assert np.allclose(Q.quat_apply(q, v)[i].numpy(), out, atol=1e-12)
        neg = np.zeros(4); mujoco.mju_negQuat(neg, q[i].numpy())
        mujoco.mju_rotVecQuat(out, v[i].numpy(), neg)
        assert np.allclose(Q.quat_apply_inverse(q, v)[i].numpy(), out, atol=1e-12)
        prod = np.zeros(4); mujoco.mju_mulQuat(prod, q[i].numpy(), p[i].numpy())
        assert np.allclose(Q.quat_mul(q, p)[i].numpy(), prod, atol=1e-12)
        m = np.zeros(9); mujoco.mju_quat2Mat(m, q[i].numpy())
        assert np.allclose(Q.matrix_from_quat(q)[i].numpy().ravel(), m, atol=1e-12)


def test_euler_round_trip_and_axis_angle():
    r, p, y = torch.tensor([0.3, -0.2]), torch.tensor([0.1, 0.4]), torch.tensor([-2.0, 1.5])
    q = Q.quat_from_euler_xyz(r, p, y)
    r2, p2, y2 = Q.euler_xyz_from_quat(q)
    for a, b in ((r, r2), (p, p2), (y, y2)):
        assert torch.allclose(Q.wrap_to_pi(a - b), torch.zeros(2), atol=1e-6)
    yaw = Q.quat_from_angle_axis(torch.tensor([math.pi / 2]), torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(Q.quat_apply(yaw, torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


class _Proxy:                      # stands in for Isaac Lab's ProxyArray
    def __init__(self, t):
        self.torch = t


def test_torch_data_converts_quaternions_and_poses_only():
    class Data:
        root_quat_w = _Proxy(torch.tensor([[0.1, 0.2, 0.3, 0.9]]))                      # x, y, z, w
        body_com_pose_w = _Proxy(torch.tensor([[[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]]]))
        joint_pos = _Proxy(torch.tensor([[0.5, 0.6]]))
        force_matrix_w_history = None
    d = TorchData(owner=type("Owner", (), {"data": Data()})())
    assert torch.equal(d.root_quat_w, torch.tensor([[0.9, 0.1, 0.2, 0.3]]))
    assert torch.equal(d.body_com_pose_w, torch.tensor([[[1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]]]))
    assert torch.equal(d.joint_pos, torch.tensor([[0.5, 0.6]]))
    assert d.force_matrix_w_history is None
