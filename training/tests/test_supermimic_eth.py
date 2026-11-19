"""supermimic/eth.py: URDF forward kinematics against MuJoCo, and one step of the analytic world model."""
import os
import xml.etree.ElementTree as ET

import pytest
import torch

from supermimic.eth import ChhayaMDP, LegFK, WorldModelEth
from supermimic.world_model import Layout

HERE = os.path.dirname(__file__)
URDF = os.path.join(HERE, "..", "robot_hardware", "robot_description", "urdf", "chhaya.urdf")
MJCF = os.path.join(HERE, "..", "envs", "assets", "robotV2", "chhaya.xml")
NAMES = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
FEET = ("l_foot_link_1", "r_foot_link_1")


def test_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    fk = LegFK(URDF, NAMES)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        q = (torch.rand(1, len(NAMES), generator=g, dtype=torch.float64) - 0.5) * 1.0
        d.qpos[:] = m.qpos0
        for i, n in enumerate(NAMES):
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = float(q[0, i])
        mujoco.mj_kinematics(m, d)
        base = m.body("base_link").id
        R0, p0 = d.xmat[base].reshape(3, 3), d.xpos[base]
        want = torch.tensor([R0.T @ (d.xpos[m.body(f).id] - p0) for f in FEET])
        got, n = fk(q.float(), normals=True)
        assert torch.allclose(got[0].double(), want, atol=1e-4), (got, want)
        want_n = torch.tensor([R0.T @ d.xmat[m.body(f).id].reshape(3, 3)[:, 2] for f in FEET])
        assert torch.allclose(n[0].double(), want_n, atol=1e-4), (n, want_n)


def _model():
    J, P = len(NAMES), 6
    b = P + 10
    S = b + 2 * J + 6
    ph = {"v_b": (P, P + 3), "h": (P + 3, P + 4), "q": (b, b + J), "qd": (b + J, b + 2 * J),
          "w_b": (b + 2 * J, b + 2 * J + 3), "g_b": (b + 2 * J + 3, b + 2 * J + 6), "cmd": (0, 3),
          "action_dst": [3, 4, 5], "action_src": [0, 1, 2], "joint_names": NAMES, "q0": [0.0] * J,
          "soft_lo": [-1.0] * J, "soft_hi": [1.0] * J, "dt": 0.02, "decimation": 4, "urdf": URDF}
    lay = Layout(obs=list(range(P)), obs_noise_std=[0.0] * P, action_dst=[3, 4, 5], action_src=[0, 1, 2],
                 hold=[0, 1, 2], phys=ph)
    return WorldModelEth(S, J, 2, lay, (32, 32), ChhayaMDP(ph, "cpu")), ph, S, J


def test_eth_step_integrates_and_differentiates():
    wm, ph, S, J = _model()
    s = torch.randn(4, S) * 0.1
    s[:, ph["g_b"][0]:ph["g_b"][1]] = torch.tensor([0.0, 0.0, -1.0])       # upright
    s[:, ph["h"][0]] = 0.28
    a = torch.zeros(4, J, requires_grad=True)
    s2, r, logit = wm(s, a, torch.zeros(4, 2))
    assert s2.shape == s.shape and r.shape == (4,) and torch.isfinite(r).all()
    assert (logit < 0).all()                                               # upright: survival
    q, qd, qd2 = (s[:, slice(*ph["q"])], s[:, slice(*ph["qd"])], s2[:, slice(*ph["qd"])])
    assert torch.allclose(s2[:, slice(*ph["q"])], q + 0.02 * (qd + 0.625 * (qd2 - qd)), atol=1e-6)
    r.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0                   # gradient reaches the action
    tilted = s2.detach().clone()
    tilted[:, slice(*ph["g_b"])] = torch.tensor([0.9, 0.0, -0.436])         # ~64 deg tilt
    assert (wm.mdp.term_logit(tilted) > 0).all()
    zero = torch.zeros(1, S); zero[:, ph["g_b"][1] - 1] = -1.0
    wm.mdp.feet(zero)
    assert wm.mdp._roll.abs().max() < 1e-5                                 # home pose, upright: soles flat
