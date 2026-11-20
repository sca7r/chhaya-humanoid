"""Get-up export (format 7) end to end, and the MuJoCo get-up gate on the exported bundle. A tiny fake checkpoint (the
exporter rebuilds the actor from tensor shapes only) in a temporary run folder with chhaya_getup.yaml beside it.
Walking export rules (v2-v6, the 45 deg tilt fault, the +g accelerometer check) must be unchanged. CPU only.
Run from training/:  python -m pytest tests/test_getup_export.py -q
"""
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.isaaclab import export_onnx as X  # noqa: E402

CFG = os.path.join(ROOT, "config", "chhaya_getup.yaml")
WALK_CFG = os.path.join(ROOT, "config", "chhaya.yaml")


def test_format_version_and_role_detection():
    walk = {"action_map": "residual", "residual": {"scale_neg_rad": None}}
    assert X._format_version(walk, {"mode": "cadence_grid"}) == 5            # walking: unchanged
    assert X._format_version(walk, {"mode": "cadence_grid"}, "walk") == 5
    assert X._format_version(walk, {"mode": "none"}, "getup") == 7
    assert X._load_getup("/nonexistent", CFG)["handoff"]["hold_s"] == 0.5
    assert X._load_getup("/nonexistent", WALK_CFG) is None


def test_joint_limits_come_from_the_run_config():
    rows = np.asarray(X._run_joint_limits_deg(CFG))
    assert rows.shape == (12, 2) and np.allclose(rows[6], [-108.0, 5.0]) and np.allclose(rows[2], [-22.5, 81.0])
    assert X._run_joint_limits_deg(WALK_CFG) is None
    env = X._gather_env_constants(CFG)
    assert math.degrees(env["joint_limits_min"][6]) == pytest.approx(-108.0)
    walk = X._gather_env_constants(WALK_CFG)
    assert math.degrees(walk["joint_limits_min"][6]) == pytest.approx(-120.0)   # the walking box


def test_policy_contract_resolves_box_scales():
    c = X._load_policy_contract("/nonexistent", CFG, None)
    r = c["residual"]
    assert r["scale_neg_rad"] is not None                                        # asymmetric: v6-style keys
    home, sp, sn = map(np.asarray, (r["home_rad"], r["scale_rad"], r["scale_neg_rad"]))
    rows = np.radians(np.asarray(X._run_joint_limits_deg(CFG)))
    assert np.allclose(home + sp, rows[:, 1], atol=2e-4) and np.allclose(home - sn, rows[:, 0], atol=2e-4)


def _fake_run(tmp, obs_dim=250, act=12, hidden=(32, 16)):
    run = os.path.join(tmp, "checkpoints_getup", "BipedV2_SAC_getup_test")
    os.makedirs(run)
    shutil.copy(CFG, os.path.join(run, "config.yaml"))
    g = torch.Generator().manual_seed(0)
    sd, sizes = {}, [obs_dim, *hidden]
    for i in range(len(hidden)):
        sd[f"actor.torso.net.{2 * i}.weight"] = 0.1 * torch.randn(sizes[i + 1], sizes[i], generator=g)
        sd[f"actor.torso.net.{2 * i}.bias"] = torch.zeros(sizes[i + 1])
    sd["actor.mean_layer.0.weight"] = 0.1 * torch.randn(act, hidden[-1], generator=g)
    sd["actor.mean_layer.0.bias"] = torch.zeros(act)
    sd["actor.torso.observation_normalizer._mean"] = torch.zeros(obs_dim)
    sd["actor.torso.observation_normalizer._std"] = torch.ones(obs_dim)
    ckpt = os.path.join(run, "step_100.pt")
    torch.save(sd, ckpt)
    mean = torch.zeros(obs_dim)
    mean[200:203] = torch.tensor([4.0, 0.0, 6.0])        # lying + standing data: specific force incl. g, not +g along z
    torch.save({"actor": {"mean": mean, "var": torch.ones(obs_dim), "count": torch.tensor(1e6)}},
               os.path.join(run, "step_100_obs_scalers.pt"))
    return ckpt


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    tmp = str(tmp_path_factory.mktemp("getup"))
    ckpt = _fake_run(tmp)
    out = os.path.join(tmp, "exported_getup", "bundle")
    argv, sys.argv = sys.argv, ["export_onnx", "--checkpoint", ckpt, "--out_dir", out]     # main() parses sys.argv
    try:
        rc = X.main()
    finally:
        sys.argv = argv
    assert rc == 0
    return out


def test_getup_bundle_contents(bundle):
    dc = json.load(open(os.path.join(bundle, "deploy_config.json")))
    assert dc["format_version"] == 7 and dc["policy_role"] == "getup"
    assert dc["tilt_fault_rad"] is None and dc["gait_clock"] == {"mode": "none"}
    assert dc["obs_proprio_dim"] == 50 and dc["history_size"] == 5
    assert math.degrees(dc["joint_limits_min"][6]) == pytest.approx(-108.0)
    assert math.degrees(dc["joint_limits_max"][4]) == pytest.approx(81.0)
    assert "action_scale_pos_rad" in dc and "action_scale_neg_rad" in dc
    assert dc["start_protocol"]["history"] == "zero_filled" and dc["start_protocol"]["gait_phase_obs"] == [0.0, 0.0]
    assert set(dc["handoff"]) >= {"tilt_max_deg", "gyro_max_rads", "joint_err_max_rad", "hold_s", "home_rad"}
    assert np.allclose(dc["handoff"]["home_rad"], dc["default_joint_pos"])
    assert dc["settle"]["hold_s"] == 0.3 and dc["timeout_s"] == 10.0
    info = json.load(open(os.path.join(bundle, "checkpoint_info.json")))
    assert info["policy_role"] == "getup"


def test_getup_gate_runs_on_the_bundle(bundle, tmp_path):
    """Mechanics only: a random tiny policy will not get up; the gate must run, report and FAIL (exit 1)."""
    pytest.importorskip("mujoco")
    from src.mujoco import getup_gate as G
    from utils.fall_bank import build_bank, categorize, projected_gravity, save_bank
    # a synthetic gate bank: supine / prone / left / right at the home pose, all in the gate split
    quats = []
    for axis, ang in (((0, 1, 0), -math.pi / 2), ((0, 1, 0), math.pi / 2), ((1, 0, 0), -math.pi / 2), ((1, 0, 0), math.pi / 2)):
        ax = np.asarray(axis, float)
        quats.append(np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * ax]))
    q = np.asarray(quats)
    home = np.asarray(json.load(open(os.path.join(bundle, "deploy_config.json")))["default_joint_pos"])
    n = len(q)
    states = {"root_z": np.full(n, 0.06), "root_quat": q, "joint_pos": np.tile(home, (n, 1)),
              "joint_vel": np.zeros((n, 12)), "root_lin_vel_b": np.zeros((n, 3)), "root_ang_vel_b": np.zeros((n, 3)),
              "category": categorize(projected_gravity(q)), "source": np.zeros(n, np.int8)}
    bank_path = str(tmp_path / "bank.npz")
    save_bank(build_bank(states, {}, gate_frac=1.0), bank_path)
    dc_path = os.path.join(bundle, "deploy_config.json")
    dc = json.load(open(dc_path))
    dc["timeout_s"] = 0.5                                     # short episodes: this checks the plumbing
    json.dump(dc, open(dc_path, "w"))
    report = str(tmp_path / "gate.json")
    rc = G.main(["--bundle", bundle, "--bank", bank_path, "--per_category", "1", "--conditions", "nominal",
                 "--out", report])
    rep = json.load(open(report))
    assert rc == 1 and rep["pass"] is False
    s = rep["summary"]["nominal"]
    assert s["n"] == 8 and set(s["per_category"]) == {"prone", "supine", "left_side", "right_side"}
    assert all(r["category"] in s["per_category"] for r in rep["results"]["nominal"])


def test_walking_policy_gate_refuses_a_getup_bundle(bundle):
    pytest.importorskip("mujoco")
    from src.mujoco import policy_gate as PG
    assert PG.main(["--bundle", bundle]) == 1
