"""Get-up reward kernels (envs/isaaclab/rewards/getup_rewards.py) and their wiring: values and orderings, mirror
invariance, gating, the overload integrator, the sign guard, the config's weights vs the env's implemented terms.
CPU only, no Isaac (getup_env.py is read as source). Run from training/:  python -m pytest tests/test_getup_rewards.py -q
"""
import ast
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from envs.isaaclab.rewards import getup_rewards as GR  # noqa: E402
from envs.isaaclab.rewards import rewards as R  # noqa: E402

PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]       # V2 mirror: pure L <-> R permutation (signs +1)


def _grav(axis, angle):
    """Projected gravity of a base rotated by angle about axis (body x, y or z)."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "y":      # pitch: g_b = (sin, 0, -cos)
        return torch.tensor([[s, 0.0, -c]])
    if axis == "x":      # roll: g_b = (0, -sin, -cos)
        return torch.tensor([[0.0, -s, -c]])
    return torch.tensor([[0.0, 0.0, -1.0]])


def test_upright_orders_upright_side_inverted():
    up = GR.getup_upright(_grav("z", 0.0))
    side = GR.getup_upright(_grav("x", math.pi / 2))
    back = GR.getup_upright(_grav("y", -math.pi / 2))
    inv = GR.getup_upright(_grav("x", math.pi))
    assert up.item() == pytest.approx(1.0)
    assert side.item() == pytest.approx(0.25) and back.item() == pytest.approx(0.25)
    assert inv.item() == pytest.approx(0.0, abs=1e-12)
    # monotone in tilt: a gradient toward upright from every pose
    vals = [GR.getup_upright(_grav("y", a)).item() for a in torch.linspace(0, math.pi, 13).tolist()]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_walking_upright_is_fooled_by_upside_down_but_getup_is_not():
    inv = _grav("x", math.pi)
    assert R.upright(inv).item() == pytest.approx(1.0)       # why the get-up env does not reuse it
    assert GR.getup_upright(inv).item() == pytest.approx(0.0, abs=1e-12)


def test_height_progress_and_gate():
    h = torch.tensor([0.03, 0.05, 0.17, 0.29, 0.40])
    hh = GR.height_progress(h, 0.05, 0.29)
    assert torch.allclose(hh, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    # height only pays upright: same height, upside down = 0, on the side = half
    one = torch.ones(1)
    assert GR.getup_height(one, _grav("z", 0)).item() == pytest.approx(1.0)
    assert GR.getup_height(one, _grav("x", math.pi / 2)).item() == pytest.approx(0.5)
    assert GR.getup_height(one, _grav("x", math.pi)).item() == pytest.approx(0.0, abs=1e-12)


def test_standing_needs_height_tilt_and_both_feet():
    up1, both = torch.ones(4), torch.ones(4, 2, dtype=torch.bool)
    hh = torch.tensor([0.95, 0.85, 0.95, 0.95])
    up = torch.tensor([1.0, 1.0, math.cos(math.radians(20)), 1.0])
    feet = both.clone()
    feet[3, 1] = False
    m = GR.standing_mask(hh, up, feet, 0.9, 15.0)
    assert m.tolist() == [True, False, False, False]
    assert GR.standing_mask(hh[:1], up1[:1], both[:1], 0.9, 15.0).item()


def test_stand_bonus_only_when_standing_and_peaks_at_home():
    home = torch.zeros(12)
    q = torch.zeros(2, 12)
    st = torch.tensor([True, False])
    out = GR.getup_stand(st, GR.pose_quality(q, home, 0.25, 2.0))
    assert out.tolist() == [1.0, 0.0]
    q2 = torch.full((1, 12), 0.2)                            # 0.2 rad on every joint: sum 0.48 rad^2
    expect = 0.5 * math.exp(-0.48 / 0.25) + 0.5 * math.exp(-0.48 / 2.0)
    assert GR.getup_stand(st[:1], GR.pose_quality(q2, home, 0.25, 2.0)).item() == pytest.approx(expect)


def test_still_penalty_only_when_standing():
    v = torch.ones(2, 3)
    st = torch.tensor([True, False])
    out = GR.getup_still(st, v, v, torch.ones(2, 12))
    assert out[1].item() == 0.0 and out[0].item() == pytest.approx(3 + 0.3 + 0.12)


def test_overload_integrator():
    dt, stall = 0.02, 2.94
    z = torch.zeros(1, 12)
    low = torch.full((1, 12), 0.5 * stall)
    for _ in range(100):
        z = GR.overload_step(z, low, stall, 0.7, 1.0, dt)
    assert torch.all(z == 0.0)                               # below the threshold: free
    high = torch.zeros(1, 12)
    high[0, 3] = stall                                       # one servo at stall for 1 s
    for _ in range(50):
        z = GR.overload_step(z, high, stall, 0.7, 1.0, dt)
    expect = 0.3 * 1.0 * (1 - math.exp(-1.0))               # dt * 0.3 per step, leaking with tau = 1 s
    assert z[0, 3].item() == pytest.approx(expect, rel=0.03)
    assert GR.servo_overload(z).item() == pytest.approx(z[0, 3].item())
    for _ in range(500):                                     # it recovers once the load is off
        z = GR.overload_step(z, low, stall, 0.7, 1.0, dt)
    assert z[0, 3].item() < 1e-4


def test_impact_and_rate_dead_zones():
    w = 1.66 * 9.81
    f = torch.tensor([0.5 * w, 2.0 * w, 3.0 * w, 100.0 * w])
    out = GR.body_impact(f, w, 2.0)
    assert out[0] == 0 and out[1] == 0 and out[2].item() == pytest.approx(1.0) and out[3].item() == 25.0
    wv = torch.tensor([[0.0, 3.9, 0.0], [5.0, 0.0, 0.0]])
    out = GR.ang_vel_limit(wv, 4.0)
    assert out[0] == 0 and out[1].item() == pytest.approx(1.0)


def test_stand_score_has_no_cliff_and_matches_the_mask():
    up = torch.ones(5)
    feet = torch.ones(5, 2, dtype=torch.bool)
    hh = torch.tensor([0.6, 0.75, 0.85, 0.9, 1.0])
    sc = GR.stand_score(hh, up, feet, 0.7, 0.9, math.cos(math.radians(30)), math.cos(math.radians(15)))
    assert sc.tolist()[0] == 0.0 and sc[3].item() == pytest.approx(1.0) and sc[4].item() == pytest.approx(1.0)
    assert 0.0 < sc[1] < sc[2] < 1.0                         # graded below the mask threshold: a way back up
    mask = GR.standing_mask(hh, up, feet, 0.9, 15.0)
    assert torch.all(sc[mask] == 1.0)
    one_foot = feet.clone()
    one_foot[:, 1] = False
    assert GR.stand_score(hh, up, one_foot, 0.7, 0.9, 0.866, 0.966)[4].item() == pytest.approx(0.5)


def test_pose_quality_keeps_a_gradient_far_from_home():
    home = torch.zeros(12)
    e2 = torch.tensor([0.0, 0.04, 0.5, 1.1, 2.0, 4.4])      # rad^2: home ... run 1's wide stance (1.1), crouch (4.4)
    q = torch.sqrt(e2 / 12).unsqueeze(-1).expand(-1, 12)
    qual = GR.pose_quality(q, home, 0.25, 2.0)
    assert qual[0].item() == pytest.approx(1.0)
    assert torch.all(qual[1:] < qual[:-1])                   # strictly decreasing: always a pull toward the home
    old = torch.exp(-e2 / 0.5)                               # the 2026-11-01 kernel: flat out there
    assert old[-1] < 2e-4 and qual[-1] > 0.05
    assert (qual[4] - qual[5]).item() > 5 * (old[4] - old[5]).item()    # crouch -> mid: the old kernel gave ~nothing


def _soles(deg):
    a = math.radians(deg)
    return torch.tensor([[[math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]] * 2])     # both soles rolled deg


def test_foot_flat_quality_has_no_dead_zone():
    root = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    feet = torch.ones(1, 2, dtype=torch.bool)
    vals = [GR.foot_flat_quality(_soles(d), root, feet, 0.35).item() for d in (0.0, 1.0, 5.0, 20.0)]
    assert vals[0] == pytest.approx(1.0)
    assert vals[0] - vals[1] > 0.08                          # L1 slope at flat (sin^2 would give ~0 here)
    assert vals[2] == pytest.approx(math.exp(-2 * math.sin(math.radians(5)) / 0.35), rel=1e-4)
    assert vals[3] < 0.15
    lifted = torch.zeros(1, 2, dtype=torch.bool)             # unloaded feet are not judged
    assert GR.foot_flat_quality(_soles(20.0), root, lifted, 0.35).item() == pytest.approx(1.0)


def _config():
    import yaml
    with open(os.path.join(ROOT, "config", "config_getup.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["train"]["env_config"]


def _task_reward(w, gc, hh, up, feet, q, feet_quat, root_quat, home):
    """The config-weighted pose-dependent task terms, held still (the regularisers depend on motion, not the pose)."""
    grav = torch.stack([torch.zeros_like(up), -torch.sqrt((1 - up ** 2).clamp(min=0)), -up], dim=-1)
    st = GR.stand_score(hh, up, feet, gc["stand_soft_height_frac"], gc["stand_height_frac"],
                        math.cos(math.radians(gc["stand_soft_tilt_deg"])), math.cos(math.radians(gc["stand_tilt_deg"])))
    pq = GR.pose_quality(q, home, gc["pose_sharp_rad2"], gc["pose_broad_rad2"])
    fq = GR.foot_flat_quality(feet_quat, root_quat, feet, gc["foot_flat_sigma"])
    return (w["getup_upright"] * GR.getup_upright(grav) + w["getup_height"] * GR.getup_height(hh, grav)
            + w["getup_feet_support"] * GR.getup_feet_support(grav, feet, gc["feet_support_min_up"])
            + w["getup_stand"] * GR.getup_stand(st, pq) + w["getup_feet_flat"] * GR.getup_feet_flat(st, fq))


def test_rising_never_lowers_the_reward_whatever_the_pose():
    """No valley across the rise. Run 2026-11-02_11-45-46 squatted at h_hat ~0.55 for 233M steps: posture PENALTIES
    gated in from h_hat 0.6 made the climb (joints far from home) score below the squat. For random poses, sole rolls,
    uprightness and foot loading held fixed, the config-weighted reward must be non-decreasing in h_hat."""
    ec = _config()
    w, gc = ec["reward_weights"], ec["getup"]
    g = torch.Generator().manual_seed(1)
    n, hs = 512, torch.linspace(0.0, 1.0, 41)
    home = torch.zeros(12)
    q = 1.2 * torch.randn(n, 12, generator=g)                # far stances included
    up = 0.6 + 0.4 * torch.rand(n, generator=g)
    feet = torch.rand(n, 2, generator=g) > 0.3
    feet_quat = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0]) + 0.4 * torch.randn(n, 2, 4, generator=g),
                                              dim=-1)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4)
    rs = torch.stack([_task_reward(w, gc, torch.full((n,), h.item()), up, feet, q, feet_quat, root_quat, home)
                      for h in hs], dim=-1)                  # (n, H)
    assert torch.all(rs[:, 1:] >= rs[:, :-1] - 1e-6)
    # the home pose on flat soles beats run 1's wide edge stance by a wide margin, which beats a low squat
    one, both = torch.ones(1), torch.ones(1, 2, dtype=torch.bool)
    r1 = lambda hh, e2, deg: _task_reward(w, gc, hh * one, one, both, torch.full((1, 12), math.sqrt(e2 / 12)),
                                          _soles(deg), root_quat[:1], home).item()
    home_r, wide, squat = r1(1.0, 0.0, 0.0), r1(1.0, 1.1, 21.0), r1(0.55, 2.5, 21.0)
    assert home_r == pytest.approx(sum(w[k] for k in ("getup_upright", "getup_height", "getup_feet_support",
                                                      "getup_stand", "getup_feet_flat")))
    assert home_r - wide > 3.0 and wide > squat


def _mirror(state):
    """The env's sagittal mirror on the reward inputs: gravity / lin vel flip y, ang vel flips x and z, joints and feet
    swap sides (getup_env reads [right, left] feet; the mirror swaps them)."""
    return {"grav": state["grav"] * torch.tensor([1.0, -1.0, 1.0]),
            "lin": state["lin"] * torch.tensor([1.0, -1.0, 1.0]),
            "ang": state["ang"] * torch.tensor([-1.0, 1.0, -1.0]),
            "q": state["q"][:, PERM], "qd": state["qd"][:, PERM], "tau": state["tau"][:, PERM],
            "feet": state["feet"][:, [1, 0]], "h": state["h"], "base_f": state["base_f"],
            "feet_quat": _quat_mirror(state["feet_quat"])[:, [1, 0]], "root_quat": _quat_mirror(state["root_quat"]),
            "self_f": state["self_f"].flip(-1)}


def _quat_mirror(q):
    return q * torch.tensor([1.0, -1.0, 1.0, -1.0])


def _all_terms(s, home):
    up = GR.uprightness(s["grav"])
    hh = GR.height_progress(s["h"], 0.05, 0.286)
    st = GR.stand_score(hh, up, s["feet"], 0.7, 0.9, math.cos(math.radians(30)), math.cos(math.radians(15)))
    pq = GR.pose_quality(s["q"], home, 0.25, 2.0)
    fq = GR.foot_flat_quality(s["feet_quat"], s["root_quat"], s["feet"], 0.35)
    z = GR.overload_step(torch.zeros_like(s["tau"]), s["tau"], 2.94, 0.7, 1.0, 0.02)
    return {
        "getup_upright": GR.getup_upright(s["grav"]), "getup_height": GR.getup_height(hh, s["grav"]),
        "getup_feet_support": GR.getup_feet_support(s["grav"], s["feet"], 0.7),
        "getup_stand": GR.getup_stand(st, pq), "getup_feet_flat": GR.getup_feet_flat(st, fq),
        "getup_still": GR.getup_still(st * pq, s["lin"], s["ang"], s["qd"]),
        "servo_overload": GR.servo_overload(z), "body_impact": GR.body_impact(s["base_f"], 16.3, 2.0),
        "ang_vel_limit": GR.ang_vel_limit(s["ang"], 4.0), "self_contact_l2": GR.self_contact_l2(s["self_f"]),
    }


def test_every_getup_term_is_mirror_invariant():
    g = torch.Generator().manual_seed(0)
    n = 256
    grav = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    grav[: n // 2, 2] = -grav[: n // 2, 2].abs()            # half of them upright-ish so standing/stand terms fire
    s = {"grav": grav, "lin": 0.3 * torch.randn(n, 3, generator=g), "ang": 3 * torch.randn(n, 3, generator=g),
         "q": 0.3 * torch.randn(n, 12, generator=g), "qd": torch.randn(n, 12, generator=g),
         "tau": 3 * torch.randn(n, 12, generator=g), "feet": torch.rand(n, 2, generator=g) > 0.3,
         "h": 0.3 * torch.rand(n, generator=g), "base_f": 60 * torch.rand(n, generator=g),
         "self_f": 8 * torch.rand(n, 10, generator=g),
         "feet_quat": torch.nn.functional.normalize(torch.randn(n, 2, 4, generator=g), dim=-1),
         "root_quat": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)}
    # a quarter standing: near-vertical, at full height, both feet down, so the gated terms (stand, still) fire
    k = n // 4
    s["grav"][:k] = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, -1.0]) + 0.1 * torch.randn(k, 3, generator=g), dim=-1)
    s["h"][:k], s["feet"][:k] = 0.29, True
    home = torch.zeros(12)                                   # the residual home is L/R symmetric (check_residual_map)
    a, b = _all_terms(s, home), _all_terms(_mirror(s), home)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    assert (a["getup_stand"] > 0).any() and (a["getup_feet_flat"] > 0).any() and (a["getup_still"] > 0).any()


def test_success_reward_is_the_value_of_standing_at_home_forever():
    """getup_success (paid on the terminating step) = r_home x gamma / (1 - gamma), r_home = the per-step task reward
    at the home pose on flat soles (every positive dense weight). Then reaching the hand-off is worth exactly what an
    endless perfect hold would be, so success never pays less than any non-terminal state can (all score <= r_home
    per step), and the Q scale stays that of the old time-out-bootstrapped episodes."""
    import yaml
    with open(os.path.join(ROOT, "config", "config_getup.yaml"), encoding="utf-8") as fh:
        full = yaml.safe_load(fh)
    w, gamma = full["train"]["env_config"]["reward_weights"], full["buffer"]["discount_factor"]
    r_home = sum(v for k, v in w.items() if v > 0 and k != "getup_success")
    assert r_home == pytest.approx(8.0)
    assert w["getup_success"] == pytest.approx(r_home * gamma / (1.0 - gamma), rel=1e-6)
    assert full["train"]["env_config"]["getup"]["terminate_on_success"] is True


def _getup_env_src():
    with open(os.path.join(ROOT, "envs", "isaaclab", "getup_env.py"), encoding="utf-8") as fh:
        return fh.read()


def _literal(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_env_implements_exactly_its_declared_terms():
    src = _getup_env_src()
    declared = set(_literal(src, "GETUP_REWARD_TERMS"))
    tree = ast.parse(src)
    raw = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "raw" for t in node.targets):
            raw = {k.value for k in node.value.keys}
    assert raw == declared


def test_signs_registered_and_config_weights_valid():
    declared = set(_literal(_getup_env_src(), "GETUP_REWARD_TERMS"))
    assert declared <= set(R.REWARD_WEIGHT_SIGN), declared - set(R.REWARD_WEIGHT_SIGN)
    ec = _config()
    w = ec["reward_weights"]
    assert set(w) <= declared, set(w) - declared             # no walking term leaks in (the env would raise)
    assert R.reward_weight_sign_errors(w, {"action_rate_weight_start": ec["action_rate_weight_start"],
                                           "action_rate_weight_end": ec["action_rate_weight_end"]}) == []
    # the stand bonus is the largest DENSE term (getup_success is the terminal one)
    dense = {k: v for k, v in w.items() if k != "getup_success"}
    assert max(dense, key=lambda k: dense[k]) == "getup_stand"
    # no posture PENALTY: every pose-dependent term is a bonus (test_rising_never_lowers_the_reward_whatever_the_pose)
    assert not {"getup_pose_l2", "foot_flat_l1", "foot_flat_l2"} & set(w)
