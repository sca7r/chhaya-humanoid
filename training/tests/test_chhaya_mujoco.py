"""Chhaya in the MuJoCo eval env (BipedEnvV2 with env_config.arms.enabled): same
policy interface as the legs-only robot, arms servo-driven toward scripted targets, stands with the
home pose, and the policy gate does not count arm links as legs."""
import copy
import os

import numpy as np
import pytest

from utils import Config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XML = os.path.join(ROOT, "envs/assets/robotV2/chhaya.xml")
pytestmark = pytest.mark.skipif(not os.path.exists(XML), reason="chhaya.xml not present")


def _env(mode="rest"):
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    cfg = Config(os.path.join(ROOT, "config/chhaya.yaml"))
    ec = cfg.train.env_config
    ec.arms = {**dict(ec.arms), "mujoco_arm_mode": mode}
    return BipedEnvV2(history_size=5, use_history=True, env_config=ec, seed=0)


def _legs_env():
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    cfg = Config(os.path.join(ROOT, "config/config.yaml"))
    return BipedEnvV2(history_size=5, use_history=True, env_config=cfg.train.env_config, seed=0)


def test_same_policy_interface_as_legs_only():
    a, b = _env(), _legs_env()
    oa, _ = a.reset(seed=0)
    ob, _ = b.reset(seed=0)
    assert a.model.nu == 20 and b.model.nu == 12
    assert a.action_space.shape == b.action_space.shape == (12,)
    assert {k: v.shape for k, v in oa.items()} == {k: v.shape for k, v in ob.items()}


def test_stands_at_home_with_arms_at_rest():
    env = _env("rest")
    env.reset(seed=0)
    env.set_command(0.0, 0.0, 0.0)
    for _ in range(150):                 # 3 s, zero action = hold the home pose
        _, _, term, _, _ = env.step(np.zeros(12, np.float32))
        assert not term
    from chhaya import motions
    q = env.data.qpos[env._arm_ids]
    rest = np.array([motions.REST[n] for n in env._arm_names])
    assert np.abs(q - rest).max() < np.radians(3.0)
    assert env.data.qpos[2] > 0.2


@pytest.mark.parametrize("mode", ["swing", "random"])
def test_arms_move_in_motion_modes_without_falling(mode):
    env = _env(mode)
    env.reset(seed=1)
    env.set_command(0.1, 0.0, 0.0)
    qs = []
    for _ in range(100):
        _, _, term, _, _ = env.step(np.zeros(12, np.float32))
        qs.append(env.data.qpos[env._arm_ids].copy())
    qs = np.array(qs)
    assert np.ptp(qs, axis=0).max() > 0.1          # arms really move
    assert np.isfinite(qs).all()


def test_gate_does_not_count_arms_as_legs():
    from src.mujoco.policy_gate import leg_sides
    env = _env()
    side, stem = leg_sides(env.model)
    import mujoco
    for b in range(env.model.nbody):
        n = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if any(k in n for k in ("shoulder", "upper_arm", "elbow", "forearm")):
            assert side[b] == -1, n
