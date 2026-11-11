"""ArmMotion (envs/isaaclab/mdp/arm_motion.py): the arm targets the upper-body robot follows while a
legs-only policy walks. Pure torch, no Isaac Lab."""
import math

import pytest
import torch

from envs.isaaclab.mdp.arm_motion import ArmMotion, MODES, REST

NAMES = ["l_shoulder_pitch", "r_shoulder_pitch", "l_shoulder_roll", "r_shoulder_roll",
         "l_arm_yaw", "r_arm_yaw", "l_elbow", "r_elbow"]
LO = torch.tensor([-3.0, -3.0, 0.05, 0.05, -1.57, -1.57, -2.0, -2.0]) * 0.9
HI = torch.tensor([1.0, 1.0, 2.6, 2.6, 1.57, 1.57, 0.1, 0.1]) * 0.9


def make(n=512, cfg=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    am = ArmMotion(n, NAMES, LO, HI, cfg, "cpu", generator=g)
    am.reset(torch.arange(n))
    return am


def test_modes_follow_probabilities():
    am = make(4000, {"mode_probs": {"rest": 1.0, "swing": 1.0, "random": 1.0, "hold": 1.0}})
    frac = torch.bincount(am.mode, minlength=4).float() / 4000
    assert torch.allclose(frac, torch.full((4,), 0.25), atol=0.04)


def test_targets_stay_in_limits_and_rate_limited():
    am = make(256, {"max_speed_rad_s": 4.0})
    dt, prev = 0.02, am.targets.clone()
    for k in range(500):
        t = am.step(dt, torch.full((256,), (k * 0.1) % (2 * math.pi)))
        assert (t >= LO - 1e-6).all() and (t <= HI + 1e-6).all()
        assert ((t - prev).abs() <= 4.0 * dt + 1e-5).all()
        prev = t.clone()


def test_rest_mode_holds_rest_pose_plus_offset():
    am = make(64, {"mode_probs": {"rest": 1.0, "swing": 0, "random": 0, "hold": 0}, "rest_offset_rad": 0.1})
    for _ in range(100):
        t = am.step(0.02, torch.zeros(64))
    rest = torch.tensor([REST[n[2:]] for n in NAMES])
    assert ((t - rest).abs() <= 0.1 + 1e-5).all()
    assert (t[:, 2:4] > 0.0).all()          # arms never swing into the legs (roll stays positive)


def test_swing_is_antiphase_and_follows_gait_clock():
    am = make(32, {"mode_probs": {"rest": 0, "swing": 1.0, "random": 0, "hold": 0}, "rest_offset_rad": 0.0,
                   "swing_amp_range": [0.4, 0.4]})
    lp, rp = [], []
    for k in range(400):
        ph = torch.full((32,), (2 * math.pi * 1.0 * k * 0.02) % (2 * math.pi))
        t = am.step(0.02, ph)
        if k > 100:
            lp.append(t[0, 0].item()); rp.append(t[0, 1].item())
    lp, rp = torch.tensor(lp), torch.tensor(rp)
    assert lp.max() - lp.min() > 0.6                       # ~ +-0.4 rad swing
    assert torch.corrcoef(torch.stack([lp, rp]))[0, 1] < -0.9   # left and right in anti-phase


def test_reset_only_touches_given_envs():
    am = make(8)
    for _ in range(50):
        am.step(0.02, torch.zeros(8))
    before = am.targets.clone()
    am.reset(torch.tensor([3]))
    assert torch.equal(am.targets[[0, 1, 2, 4, 5, 6, 7]], before[[0, 1, 2, 4, 5, 6, 7]])


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        make(4, {"mode_probs": {"dance": 1.0}})
    assert set(MODES) == {"rest", "swing", "random", "hold"}
