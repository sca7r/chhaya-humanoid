"""Whole-body policy (env_config.arms.control: policy): the Isaac env's arm-block logic, run on the CPU.

Isaac Lab cannot be imported here, so the REAL methods are lifted out of envs/isaaclab/biped_env.py by AST
(as tests/test_symmetric_obs_stats.py does) and run on a stand-in object."""
import ast
import os
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARMS = ["l_shoulder_pitch", "r_shoulder_pitch", "l_shoulder_roll", "r_shoulder_roll",
        "l_arm_yaw", "r_arm_yaw", "l_elbow", "r_elbow"]
SCALE = {"shoulder_pitch": 1.0, "shoulder_roll": 0.6, "arm_yaw": 0.8, "elbow": 1.0}
P, HIST = 74, 5


def _lift(*names):
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in fns} == set(names)
    ns = {"torch": torch}
    exec(compile(ast.Module(fns, []), "biped_env_methods", "exec"), ns)
    return {n: ns[n] for n in names}


class Fake:
    device = torch.device("cpu")
    num_envs = 4
    obs_proprio_dim = P
    _n_arm_act = 8
    EXTRA_CRITIC_DIM = 0
    num_obs_policy = P * HIST
    rd = property(lambda self: self.robot.data)     # BipedEnv.rd: robot data as tensors (compat.TorchData)

    def __init__(self):
        leg_default = torch.zeros(12)
        arm_default = torch.tensor([0.0, 0.0, 0.20, 0.20, 0.0, 0.0, -0.35, -0.35])
        self._arm_list = list(range(12, 20))
        self.robot = SimpleNamespace(data=SimpleNamespace(
            default_joint_pos=torch.cat([leg_default, arm_default]).unsqueeze(0).repeat(self.num_envs, 1)))
        self.cfg = SimpleNamespace(arm_action_scale_rad=SCALE, arm_mirror_signs=None,
                                   mirror_joint_perm=[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
                                   mirror_joint_signs=[1.0] * 12, policy_has_privileged_info=False)
        lim = torch.tensor([[-3.0, 1.0]] * 2 + [[0.05, 2.6]] * 2 + [[-1.57, 1.57]] * 2 + [[-2.0, 0.1]] * 2)
        self._setup_arm_policy(ARMS, lim)
        self._build_mirror_transform()


for _n, _f in _lift("_setup_arm_policy", "arm_actions_to_targets", "_build_mirror_transform", "mirror_obs",
                    "mirror_action", "actor_mirror_tables").items():
    setattr(Fake, _n, _f)


def test_arm_action_map():
    e = Fake()
    assert torch.allclose(e.arm_actions_to_targets(torch.zeros(4, 8)), e.arm_default.expand(4, 8))
    hi = e.arm_actions_to_targets(torch.ones(4, 8))[0]
    lo = e.arm_actions_to_targets(-torch.ones(4, 8))[0]
    assert torch.allclose(hi, torch.tensor([1.0, 1.0, 0.80, 0.80, 0.8, 0.8, 0.1, 0.1]))     # elbow clamps at +0.1
    assert torch.allclose(lo, torch.tensor([-1.0, -1.0, 0.05, 0.05, -0.8, -0.8, -1.35, -1.35]))  # roll clamps at 0.05
    assert torch.allclose(e.arm_actions_to_targets(5 * torch.ones(4, 8)), hi.expand(4, 8))   # action is clipped


def test_arm_joint_order_is_checked():
    e = Fake()
    with pytest.raises(ValueError):
        e._setup_arm_policy(ARMS[::-1], torch.zeros(8, 2))


def test_mirror_is_an_involution_and_swaps_the_arm_blocks():
    e = Fake()
    obs = {"policy": torch.randn(4, P * HIST), "critic": torch.randn(4, P + 10)}
    m = e.mirror_obs(obs)
    mm = e.mirror_obs(m)
    assert torch.allclose(mm["policy"], obs["policy"]) and torch.allclose(mm["critic"], obs["critic"])
    newest, mnew = obs["policy"][:, -P:], m["policy"][:, -P:]
    for k in range(3):                                   # arm joint_pos_rel, joint_vel, previous actions
        lo = 50 + 8 * k
        assert torch.allclose(mnew[:, lo:lo + 8:2], newest[:, lo + 1:lo + 8:2])      # l <- r
        assert torch.allclose(mnew[:, lo + 1:lo + 8:2], newest[:, lo:lo + 8:2])      # r <- l
    assert torch.allclose(mnew[:, 1], -newest[:, 1])                                 # leg frame untouched: acc y flips
    # privileged block sits after the 74-dim frame: lin_vel y flips, the two feet's forces swap
    c, mc = obs["critic"], m["critic"]
    assert torch.allclose(mc[:, P + 1], -c[:, P + 1])
    assert torch.allclose(mc[:, P + 4], c[:, P + 7]) and torch.allclose(mc[:, P + 5], -c[:, P + 8])


def test_action_mirror_is_20_wide():
    e = Fake()
    a = torch.randn(4, 20)
    m = e.mirror_action(a)
    assert torch.allclose(e.mirror_action(m), a)
    assert torch.allclose(m[:, 12::2], a[:, 13::2]) and torch.allclose(m[:, 0], a[:, 1])
    perm, sign, aperm, asign = e.actor_mirror_tables()
    assert perm.shape == (P * HIST,) and aperm.shape == (20,)
