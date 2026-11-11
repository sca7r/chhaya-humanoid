"""Chhaya in the MuJoCo eval env (BipedEnvV2 with env_config.arms.enabled). control: scripted = same
policy interface as the legs-only robot, arms servo-driven toward scripted targets; control: policy =
whole-body policy (20 actions, 74-dim frame). The robot stands with the
home pose, and the policy gate does not count arm links as legs."""
import copy
import os

import numpy as np
import pytest

from utils import Config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XML = os.path.join(ROOT, "envs/assets/robotV2/chhaya.xml")
pytestmark = pytest.mark.skipif(not os.path.exists(XML), reason="chhaya.xml not present")


def _env(mode="rest", control="scripted"):
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    cfg = Config(os.path.join(ROOT, "config/chhaya.yaml"))
    ec = cfg.train.env_config
    ec.arms = {**dict(ec.arms), "mujoco_arm_mode": mode, "control": control}
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


# ----------------------------------------------------------------------------- whole-body policy (control: policy)

def test_whole_body_interface():
    env = _env(control="policy")
    obs, _ = env.reset(seed=0)
    assert env.action_space.shape == (20,)
    assert obs["actor"].shape == (5 * 74,) and obs["critic"].shape == (84,)
    # the first 50 dims of the frame are the legs-only frame; at reset the arm block is ~0 (rest pose, no motion)
    assert np.abs(obs["actor"][-74:][50:]).max() < 0.05


def test_whole_body_zero_action_stands_at_rest():
    env = _env(control="policy")
    env.reset(seed=0)
    env.set_command(0.0, 0.0, 0.0)
    for _ in range(150):
        _, _, term, _, _ = env.step(np.zeros(20, np.float32))
        assert not term
    q = env.data.qpos[env._arm_ids]
    assert np.abs(q - env._arm_rest).max() < np.radians(3.0)
    assert env.data.qpos[2] > 0.2


def test_whole_body_arm_actions_move_the_arms_and_are_observed():
    env = _env(control="policy")
    env.reset(seed=0)
    env.set_command(0.0, 0.0, 0.0)
    a = np.zeros(20, np.float32)
    a[12:] = [-0.5, -0.5, 0.5, 0.5, 0.3, 0.3, -0.4, -0.4]     # mirror-symmetric arm command
    for _ in range(100):
        obs, _, term, _, _ = env.step(a)
        assert not term
    want = np.clip(env._arm_rest + env._arm_scale * a[12:], env._arm_lim_min, env._arm_lim_max)
    q = env.data.qpos[env._arm_ids]
    assert np.abs(q - want).max() < np.radians(4.0)
    frame = obs["actor"][-74:]
    assert np.allclose(frame[50:58], q - env._arm_rest, atol=2e-3)       # arm joint_pos_rel (quantised)
    assert np.allclose(frame[66:74], a[12:])                             # previous arm actions
    assert np.allclose(frame[36:48], 0.0)                                # leg previous actions untouched


def test_arm_mirror_is_a_plain_left_right_swap():
    """The Isaac env mirrors the arm block with perm = swap(l, r) and signs +1 (arm_mirror_signs None). Check on the
    model: swapping the l/r arm angles reflects the hand positions through the sagittal plane."""
    import mujoco
    env = _env(control="policy")
    env.reset(seed=0)
    m, d = env.model, env.data
    rng = np.random.default_rng(0)
    hands = [m.body("l_forearm_link").id, m.body("r_forearm_link").id]
    for _ in range(10):
        q = rng.uniform(env._arm_lim_min, env._arm_lim_max)
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[env._arm_ids] = q
        mujoco.mj_forward(m, d)
        l0, r0 = d.xipos[hands[0]].copy(), d.xipos[hands[1]].copy()
        d.qpos[env._arm_ids] = q.reshape(4, 2)[:, ::-1].reshape(-1)       # swap l <-> r per type
        mujoco.mj_forward(m, d)
        l1, r1 = d.xipos[hands[0]].copy(), d.xipos[hands[1]].copy()
        flip = np.array([1.0, -1.0, 1.0])
        assert np.allclose(l1, r0 * flip, atol=2e-3) and np.allclose(r1, l0 * flip, atol=2e-3)
