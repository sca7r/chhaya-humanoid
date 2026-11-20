"""Get-up task plumbing without Isaac: config_getup.yaml (joint box = URDF x 0.9, box action scales reach it exactly),
env_config's getup / joint-limit handlers, the critic mirror table with the extra dims, and the MuJoCo env's get-up
mode (zero command / phase, start from a fall-bank state with the Jetson re-arm). Run from training/:
    python -m pytest tests/test_getup_config.py -q
"""
import ast
import copy
import math
import os
import re
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import Config  # noqa: E402
from utils.residual_action import JOINT_TYPES, actions_to_targets_np, home_rad, limits_rows_deg, scale_pair_rad  # noqa: E402
from utils.residual_home import freeze_residual_map  # noqa: E402

CFG = os.path.join(ROOT, "config", "config_getup.yaml")


def _frozen():
    c = Config(CFG)
    freeze_residual_map(c)
    return c


def _src(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def _module_literal(rel, name):
    for node in ast.parse(_src(rel)).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) == name for t in targets):
                return ast.literal_eval(node.value)
    raise KeyError(name)


def _urdf_ranges_deg():
    """{joint type: (lo, hi)} from the MJCF (the URDF-derived model), left leg."""
    xml = _src(os.path.join("envs", "assets", "robotV2", "robot_v2.xml"))
    out = {}
    for t, jn in zip(JOINT_TYPES, ("l_hip_yaw", "l_hip_roll_joint", "l_hip_pitch_joint", "l_knee_joint",
                                   "l_ankle_roll_joint", "l_ankle_pitch_joint")):
        m = re.search(rf'<joint name="{jn}"[^>]*\srange="([-\d.]+) ([-\d.]+)"', xml)
        out[t] = (math.degrees(float(m.group(1))), math.degrees(float(m.group(2))))
    return out


# ---------------- 1. config ----------------

def test_joint_box_is_urdf_times_0_9_with_the_knee_stop():
    lim = Config(CFG).train.env_config.joint_limits_deg
    for t, (lo, hi) in _urdf_ranges_deg().items():
        want_hi = 5.0 if t == "knee" else 0.9 * hi       # the knee keeps its +5 deg hyperextension stop
        assert lim[t][0] == pytest.approx(0.9 * lo, abs=0.06), t
        assert lim[t][1] == pytest.approx(want_hi, abs=0.06), t


def test_box_scales_reach_the_box_edges_and_home_is_the_walking_home():
    c = _frozen()
    ec = c.train.env_config
    assert isinstance(ec.action_scale_rad, dict)               # frozen to numbers (never "box" in a run config)
    sp, sn = scale_pair_rad(ec.action_scale_rad)
    home = home_rad(ec.action_home_deg)
    lim = np.radians(limits_rows_deg(ec.joint_limits_deg))
    hi = actions_to_targets_np(np.ones(12), home, sp, lim[:, 0], lim[:, 1], sn)
    lo = actions_to_targets_np(-np.ones(12), home, sp, lim[:, 0], lim[:, 1], sn)
    assert np.allclose(hi, lim[:, 1], atol=2e-4) and np.allclose(lo, lim[:, 0], atol=2e-4)
    assert np.allclose(actions_to_targets_np(np.zeros(12), home, sp, lim[:, 0], lim[:, 1], sn), home)
    # the deployed walking run's frozen home (the hand-off pose)
    run = os.path.join(ROOT, "checkpoints", "BipedV2_SAC_gaitref_resume_2026-10-28_08-00-32", "config.yaml")
    if os.path.exists(run):
        txt = _src(run)
        for t in JOINT_TYPES:
            m = re.search(rf"action_home_deg:(?:\n\s+\w+: [-\d.]+)*?\n\s+{t}: ([-\d.]+)", txt)
            assert m and float(m.group(1)) == pytest.approx(ec.action_home_deg[t]), t


def test_outputs_are_kept_separate():
    tr = Config(CFG).train
    assert tr.log_dir == "runs_getup" and tr.checkpoint_path.rstrip("/") == "checkpoints_getup"
    assert tr.equivariant_actor is False and tr.symmetry_augmentation is True
    assert tr.use_rough_terrain is False


# ---------------- 2. env_config handlers ----------------

def _defaults():
    return _module_literal(os.path.join("envs", "isaaclab", "getup_env_cfg.py"), "GETUP_DEFAULTS")


def test_config_getup_block_has_every_default_key():
    block = Config(CFG).train.env_config.getup
    assert set(block) == set(_defaults())                     # the config is the source of truth: nothing implicit
    for k, v in _defaults().items():
        if isinstance(v, dict) and k != "start_mix":
            assert set(block[k]) == set(v), k


def test_apply_getup_merges_validates_and_disables_assist_in_play():
    from src.isaaclab import env_config as E
    ec = Config(CFG).train.env_config
    cfg = SimpleNamespace(getup=copy.deepcopy(_defaults()), episode_length_s=20.0)
    E._apply_getup(cfg, ec, training=True)
    assert cfg.getup["assist"]["enabled"] is True and cfg.episode_length_s == 10.0
    assert cfg.getup["terminate_on_success"] is True         # training: the hand-off is the goal
    assert cfg.getup["settle"] == ec.getup["settle"]
    play = SimpleNamespace(getup=copy.deepcopy(_defaults()), episode_length_s=20.0)
    E._apply_getup(play, ec, training=False)
    assert play.getup["assist"]["enabled"] is False
    assert play.getup["terminate_on_success"] is False       # play keeps running so the stand can be watched
    bad = copy.deepcopy(ec)
    bad.getup = {**ec.getup, "assist": {**ec.getup["assist"], "strength": 1.0}}
    with pytest.raises(ValueError, match="unknown key"):
        E._apply_getup(SimpleNamespace(getup=copy.deepcopy(_defaults()), episode_length_s=20.0), bad, training=True)
    with pytest.raises(ValueError, match="only for the get-up task"):
        E._apply_getup(SimpleNamespace(), ec, training=True)


def test_apply_joint_limits_sets_radian_rows():
    from src.isaaclab import env_config as E
    ec = Config(CFG).train.env_config
    cfg = SimpleNamespace(joint_limits=None)
    E._apply_joint_limits(cfg, ec)
    rows = np.asarray(cfg.joint_limits)
    assert rows.shape == (12, 2)
    assert np.allclose(np.degrees(rows[6]), [-108.0, 5.0]) and np.allclose(np.degrees(rows[4]), [-81.0, 81.0])


def test_walking_configs_are_untouched():
    for name in ("chhaya.yaml",):
        ec = Config(os.path.join(ROOT, "config", name)).train.env_config
        assert ec.joint_limits_deg is None and ec.getup is None


# ---------------- 3. critic mirror with the get-up extras ----------------

def _lift_mirror(extra):
    import torch
    src = _src(os.path.join("envs", "isaaclab", "biped_env.py"))
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BipedEnv")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_mirror_transform")
    ns = {"torch": torch}
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    self = SimpleNamespace(device="cpu", EXTRA_CRITIC_DIM=extra,
                           cfg=SimpleNamespace(mirror_joint_perm=[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
                                               mirror_joint_signs=[1.0] * 12))
    ns["_build_mirror_transform"](self)
    return self


def test_critic_mirror_extends_with_identity_dims():
    walk, getup = _lift_mirror(0), _lift_mirror(2)
    assert walk._mirror_critic_obs_perm.numel() == 60 and getup._mirror_critic_obs_perm.numel() == 62
    assert (getup._mirror_critic_obs_perm[:60] == walk._mirror_critic_obs_perm).all()
    assert getup._mirror_critic_obs_perm[60:].tolist() == [60, 61] and getup._mirror_critic_obs_signs[60:].tolist() == [1.0, 1.0]
    p, s = getup._mirror_critic_obs_perm, getup._mirror_critic_obs_signs
    assert (p[p] == __import__("torch").arange(62)).all() and (s * s[p] == 1).all()          # an involution
    # zero command and zero phase slots are mirror-invariant (why the get-up phase obs is (0, 0), not (0, 1))
    po, so = getup._mirror_obs_perm, getup._mirror_obs_signs
    assert so[48].item() == -1 and so[49].item() == -1 and po[48].item() == 48 and po[49].item() == 49


def test_getup_env_phase_obs_is_zero_and_critic_extra_dim_matches():
    src = _src(os.path.join("envs", "isaaclab", "getup_env.py"))
    assert "EXTRA_CRITIC_DIM = 2" in src
    fn = re.search(r"def _phase_obs\(self\) -> torch.Tensor:\n\s+return (.+)\n", src).group(1)
    assert fn.startswith("torch.zeros(self.num_envs, 2")
    extra = re.search(r"def _extra_critic_obs\(self\).*?return \[(.+?)\]\n", src, re.S).group(1)
    assert extra.count("unsqueeze(-1)") == 2


# ---------------- 4. MuJoCo get-up mode ----------------

def _mj_env():
    pytest.importorskip("mujoco")
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    ec = _frozen().train.env_config
    ec.backlash = {**(ec.backlash or {}), "enabled": False}
    return BipedEnvV2(history_size=5, use_history=True, critic_has_privileged_info=True, env_config=ec, seed=0)


def test_mujoco_getup_mode_start_state_and_rearm():
    env = _mj_env()
    try:
        assert np.allclose(np.degrees(env._jlim_min[6]), -108.0, atol=1e-3)
        th = -math.pi / 2                                          # supine: pitch -90 (nose up)
        q = env._residual_home.copy()
        q[6] = q[7] = math.radians(-60.0)                          # knees bent: prev action must not be 0
        st = {"root_z": 0.06, "root_quat": [math.cos(th / 2), 0.0, math.sin(th / 2), 0.0], "joint_pos": q,
              "joint_vel": np.zeros(12), "root_lin_vel_b": np.zeros(3), "root_ang_vel_b": np.zeros(3)}
        obs, _ = env.reset(seed=0, options={"start_state": st})
        f = obs["actor"][-50:]
        assert np.allclose(f[6:9], [-1.0, 0.0, 0.0], atol=1e-3)                 # supine gravity
        assert np.allclose(f[0:3], [9.81, 0.0, 0.0], atol=1e-2)                 # at-rest specific force, IMU lag seeded
        assert np.all(f[9:12] == 0.0) and np.all(f[48:50] == 0.0)               # command 0, phase (0, 0)
        assert np.allclose(env.prev_actions, env.targets_to_actions(env._jpos()), atol=1e-6)
        assert np.allclose(env._actions_to_targets(env.prev_actions), env._jpos(), atol=1e-4)   # re-arm holds the pose
        assert env.prev_actions[6] < -0.2
        for _ in range(25):
            _, _, term, _, _ = env.step(env.prev_actions.copy())
            assert not term                                                      # no tilt termination
    finally:
        env.close()


def test_walking_mujoco_env_keeps_its_limits():
    pytest.importorskip("mujoco")
    from envs.mujoco.biped_env_v2 import JLIM_MAX, JLIM_MIN, BipedEnvV2
    ec = Config(os.path.join(ROOT, "config", "chhaya.yaml")).train.env_config
    ec.backlash = {**(ec.backlash or {}), "enabled": False}
    env = BipedEnvV2(history_size=5, use_history=True, env_config=ec, seed=0)
    try:
        assert env._jlim_min is JLIM_MIN and env._jlim_max is JLIM_MAX and not env._getup
    finally:
        env.close()
