"""Chhaya Lite: the hobby-servo variant (config/chhaya_lite.yaml, actuator/params/hobby_estimates/)."""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from actuator.bam_kernel import BamServo, load_params  # noqa: E402

KGF_CM = 0.0980665


@pytest.mark.parametrize("name, volts, stall_kgcm, s_per_60", [
    ("rds3225_6v8", 6.8, 25.0, 0.13),
    ("mg996r_6v", 6.0, 11.0, 0.14),
])
def test_params_reproduce_the_datasheet(name, volts, stall_kgcm, s_per_60):
    p = load_params(f"actuator/params/hobby_estimates/{name}.json")
    assert p.v_nominal == volts
    assert math.isclose(p.stall_torque, stall_kgcm * KGF_CM, rel_tol=1e-6)
    assert math.isclose(volts / p.kt, math.radians(60) / s_per_60, rel_tol=1e-6)      # no-load speed
    # the kernel itself, stalled at a large error and at the datasheet voltage, gives the stall torque
    servo = BamServo(p, 1, 1, 0.005, kp=110.0, vin=volts, dtype=torch.float64)
    q = torch.zeros(1, 1, dtype=torch.float64)
    tau = None
    for _ in range(40):
        tau = servo.kernel.motor(q + 1.0, q, q)
    assert math.isclose(float(tau), stall_kgcm * KGF_CM, rel_tol=1e-6)


def test_sts3215_model_is_untouched():
    p = load_params("actuator/params/sts3215_12v/m6.json")
    assert p.kd_fw == 0.0 and p.v_nominal == 12.0
    assert math.isclose(p.stall_torque, 0.97 * 12.0 * p.kt / p.R)


def _env(cfg_name):
    from utils import Config
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    cfg = Config(os.path.join(ROOT, "config", cfg_name))
    return BipedEnvV2(history_size=5, use_history=True, env_config=cfg.train.env_config, seed=0)


def test_lite_env_uses_both_servo_types_and_stands():
    env = _env("chhaya_lite.yaml")
    assert env.servo.kernel.p.actuator == "rds3225_6v8"
    assert env.arm_servo.kernel.p.actuator == "mg996r_6v"
    assert float(env.servo.kernel.vin[0, 0]) == 6.8
    assert env.action_space.shape[0] == 20
    env.reset(seed=0)
    env.set_command(0.0, 0.0, 0.0)
    for _ in range(150):
        obs, _, term, _, _ = env.step(np.zeros(20, np.float32))
        assert not term, "fell while standing"
    assert np.isfinite(obs["actor"]).all() and obs["actor"].shape == (5 * 74,)
    assert env.data.qpos[2] > 0.25


def test_lite_config_differs_from_chhaya_only_where_marked():
    import yaml
    a = yaml.safe_load(open(os.path.join(ROOT, "config", "chhaya.yaml")))
    b = yaml.safe_load(open(os.path.join(ROOT, "config", "chhaya_lite.yaml")))

    def diff(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            out = []
            for k in sorted(set(x) | set(y)):
                out += diff(x.get(k), y.get(k), f"{path}.{k}" if path else k)
            return out
        return [] if x == y else [path]
    changed = set(diff(a, b))
    e = "train.env_config."
    assert changed == {
        "train.model_name",
        e + "observation_noise_std.joint_pos", e + "observation_noise_std.joint_vel", e + "mount_offset_rad",
        e + "actuator.params_file", e + "actuator.arms_params_file", e + "actuator.vin", e + "actuator.kp_fw",
        e + "actuator.kp_fw_range", e + "actuator.vin_range", e + "actuator.friction_scale_range",
        e + "actuator.stall_torque_cap_nm", e + "actuator.deadband_counts", e + "actuator.deadband_counts_range",
        e + "backlash.lash_deg_range", e + "backlash.mujoco_lash_deg", e + "backlash.impulse_cap",
    }
