"""getup_gate.py -- pre-deploy gate for an exported GET-UP bundle (format 7), in MuJoCo (sim2sim).

Starts every episode from the fall bank's held-out GATE split (states the Isaac policy never trained from), each
state together with its exact mirror, and runs the bundle the way the Jetson will:
  * re-arm at the current pose (EMA state = the pose, previous action = targets_to_actions(pose)), history zero-filled,
    command 0, gait-phase slots (0, 0);
  * the Jetson rate limiter (a = prev + clip(raw - prev, +-rate)) with the APPLIED action in the previous-action slot,
    plus `delay` extra steps of action delay;
  * success = the bundle's handoff test (tilt, gyro and every joint near the walking home, held handoff.hold_s), the
    exact rule the Jetson switches to walking on. No foot sensors or base height are used for it.

With --walk_bundle the hand-off itself is tested: on success the physics state is copied into a walking env, the
walking bundle's arming ramp (2 s smoothstep to its home) runs, then the walking policy starts with a zeroed history
and must stand 10 s at command 0 and walk 5 s at vx 0.1 without passing 45 deg tilt.

Conditions (each a separate pass over the same starts): nominal (bundle servo P, 12 V, backlash off, the config's
mujoco_ground_friction), weak (P 20, 10.5 V: the 2026-10-27 hardware fit and the sag during recoveries), lash
(2.5 deg backlash hinges), delay2 (2 extra steps), fric_lo / fric_hi (floor friction 0.3 / 0.8).

Reports per condition: success overall / per category, mirror-pair disagreement (L/R asymmetry), median
time-to-handoff, the longest continuous time any servo spent above 90% of the stall torque, peak base rate, peak base
impact force, peak inter-leg contact force, the hand-off success rate. PASS/FAIL against the thresholds in VERDICT.
Exit code 0 = PASS. The JSON report goes to data_getup/gate/<bundle>.json (nothing else is written).

    python src/mujoco/getup_gate.py --bundle exported_getup/<name> [--walk_bundle exported/<walk name>]
                                    [--per_category 8] [--conditions nominal weak lash delay2 fric_lo fric_hi]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TRL = os.path.join(_ROOT, "torch_rl_algorithms")
if _TRL not in sys.path:
    sys.path.append(_TRL)          # appended: its top-level `config` must not shadow BipedRobot's

import mujoco  # noqa: E402

from envs.mujoco.biped_env_v2 import BipedEnvV2  # noqa: E402
from src.mujoco.policy_gate import FRAME, PREV_SLOT, Policy, _run_config_path, inter_leg_contact, leg_sides  # noqa: E402
from utils import Config  # noqa: E402
from utils.fall_bank import CATEGORIES, load_bank  # noqa: E402

TILT_FAULT = 0.784
STALL_NM = 2.94
SAT_FRAC = 0.9
CONDITIONS = {
    "nominal": {},
    "weak": {"kp": 20.0, "vin": 10.5},
    "lash": {"lash_deg": 2.5},
    "delay2": {"delay": 2},
    "fric_lo": {"friction": 0.3},
    "fric_hi": {"friction": 0.8},
}
# Initial PASS bar (plan 2026-11-01). Rates are fractions.
VERDICT = {
    "nominal_success_min": 0.95, "nominal_category_min": 0.90, "weak_success_min": 0.80, "other_success_min": 0.85,
    "median_time_max_s": 4.0, "servo_sat_max_s": 1.0, "mirror_disagree_max": 0.05, "handoff_min": 0.95,
}


def _make_env(ec, hist: int, cond: dict, seed: int) -> BipedEnvV2:
    ec = copy.deepcopy(ec)
    lash = cond.get("lash_deg")
    ec.backlash = {**(ec.backlash or {}), "enabled": bool(lash), **({"mujoco_lash_deg": float(lash)} if lash else {})}
    if "friction" in cond:
        ec.mujoco_ground_friction = float(cond["friction"])
    env = BipedEnvV2(history_size=hist, use_history=True, critic_has_privileged_info=True, env_config=ec, seed=seed)
    if "kp" in cond:
        env.servo.kernel.kp[:] = float(cond["kp"])
    if "vin" in cond:
        env.servo.kernel.vin[:] = float(cond["vin"])
    return env


def _tilt_gyro(env) -> tuple[float, float]:
    g = env._world_to_body_rot() @ np.array([0.0, 0.0, -1.0])
    return math.acos(float(np.clip(-g[2], -1.0, 1.0))), float(np.linalg.norm(env.data.qvel[3:6]))


def _base_force(env) -> float:
    m, d, f6, tot = env.model, env.data, np.zeros(6), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if env._base_body_id in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += abs(float(f6[0]))
    return tot


def _jetson_start(env, hist: int, prev: np.ndarray) -> np.ndarray:
    """History zero-filled, then the first frame (the Jetson's start()); returns the actor obs."""
    env._history.clear()
    for _ in range(hist):
        env._history.append(np.zeros(FRAME, dtype=np.float32))
    env.prev_actions[:] = prev
    return env._build_obs()["actor"].copy()


def run_getup(policy: Policy, env: BipedEnvV2, start: dict, seed: int, cond: dict, rate: float,
              handoff: dict, timeout_s: float) -> tuple[dict, BipedEnvV2]:
    np.random.seed(seed)
    env.reset(seed=seed, options={"start_state": start, "yaw": float(np.random.uniform(-math.pi, math.pi))})
    o = _jetson_start(env, policy.hist, env.prev_actions.copy())
    prev = env.prev_actions.astype(np.float64).copy()
    delay = int(cond.get("delay", 0))
    fifo = deque([prev.copy()] * (1 + delay), maxlen=1 + delay)
    home = env._default_joint_pos.astype(np.float64)
    hold = max(1, int(math.ceil(float(handoff["hold_s"]) / 0.02 - 1e-6)))
    side, stem = leg_sides(env.model)
    tilt_max, gyro_max = math.radians(float(handoff["tilt_max_deg"])), float(handoff["gyro_max_rads"])
    err_max = float(handoff["joint_err_max_rad"])
    ok_run, t_hand, sat_run, sat_max, w_pk, f_pk, leg_pk, err_at = 0, None, np.zeros(12), 0.0, 0.0, 0.0, 0.0, None
    for i in range(int(round(timeout_s / 0.02))):
        raw = policy(o)
        a = prev + np.clip(raw - prev, -rate, rate) if rate > 0 else raw
        prev = a
        fifo.append(a)
        obs, _, term, _, _ = env.step(fifo[0].astype(np.float32))
        o = obs["actor"].copy()
        o[-FRAME:][PREV_SLOT] = a                    # the Jetson feeds back the APPLIED action
        env._history[-1][PREV_SLOT] = a
        env.prev_actions[:] = a
        if term:
            break
        tau = np.abs(env.data.actuator_force[env._act_ids])
        sat_run = np.where(tau > SAT_FRAC * STALL_NM, sat_run + 0.02, 0.0)
        sat_max = max(sat_max, float(sat_run.max()))
        tilt, gyro = _tilt_gyro(env)
        w_pk = max(w_pk, gyro)
        f_pk = max(f_pk, _base_force(env))
        leg_pk = max(leg_pk, sum(inter_leg_contact(env, side, stem).values()))
        err = float(np.max(np.abs(env._jpos() - home)))
        ok_run = ok_run + 1 if (tilt < tilt_max and gyro < gyro_max and err < err_max) else 0
        if ok_run >= hold:
            t_hand, err_at = (i + 1 - hold) * 0.02, err
            break
    return {"success": t_hand is not None, "t_handoff_s": t_hand, "servo_sat_s": sat_max, "peak_w_rads": w_pk,
            "base_impact_n": f_pk, "inter_leg_n": leg_pk, "err_at_handoff_rad": err_at,
            "on_feet": bool(env._in_contact.all()) if t_hand is not None else None}, env


def _copy_state(src: BipedEnvV2, dst: BipedEnvV2) -> None:
    """Physics state by joint name (free root, the 12 servos, lash hinges if both have them)."""
    ms, md = src.model, dst.model
    for j in range(md.njnt):
        name = mujoco.mj_id2name(md, mujoco.mjtObj.mjOBJ_JOINT, j)
        js = mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_JOINT, name) if name else -1
        if js < 0:
            continue
        nq = {0: 7, 1: 4}.get(int(md.jnt_type[j]), 1)
        nv = {0: 6, 1: 3}.get(int(md.jnt_type[j]), 1)
        dst.data.qpos[md.jnt_qposadr[j]:md.jnt_qposadr[j] + nq] = src.data.qpos[ms.jnt_qposadr[js]:ms.jnt_qposadr[js] + nq]
        dst.data.qvel[md.jnt_dofadr[j]:md.jnt_dofadr[j] + nv] = src.data.qvel[ms.jnt_dofadr[js]:ms.jnt_dofadr[js] + nv]
    mujoco.mj_forward(md, dst.data)


def run_handoff(genv: BipedEnvV2, wpol: Policy, wec, cond: dict, seed: int, rate: float) -> dict:
    """Arming ramp (2 s smoothstep, the Jetson's _step_arming) into the walking home, then the walking policy:
    10 s at command 0 and 5 s at vx 0.1, failing on a 45 deg tilt."""
    wenv = _make_env(wec, wpol.hist, cond, seed)
    try:
        wenv.reset(seed=seed)
        _copy_state(genv, wenv)
        wenv.filtered_targets[:] = wenv._jpos().copy()
        wenv.servo.reset()
        wenv._reset_sensor_lag()
        q0, home = wenv._jpos().astype(np.float64).copy(), wenv._default_joint_pos.astype(np.float64)
        for k in range(100):
            s = (k + 1) / 100.0
            s = s * s * (3.0 - 2.0 * s)
            wenv.step(wenv.targets_to_actions(q0 + (home - q0) * s))
            if _tilt_gyro(wenv)[0] > TILT_FAULT:
                return {"ok": False, "phase": "arming"}
        wenv.gait_phase = 0.0
        wenv.set_command(0.0, 0.0, 0.0)
        o = _jetson_start(wenv, wpol.hist, np.zeros(12, np.float32))
        prev = np.zeros(12)
        fifo = deque([prev.copy()] * (1 + int(cond.get("delay", 0))), maxlen=1 + int(cond.get("delay", 0)))
        vx = []
        for i in range(750):
            if i == 500:
                wenv.set_command(0.1, 0.0, 0.0)
            raw = wpol(o)
            a = prev + np.clip(raw - prev, -rate, rate) if rate > 0 else raw
            prev = a
            fifo.append(a)
            obs, _, _, _, _ = wenv.step(fifo[0].astype(np.float32))
            o = obs["actor"].copy()
            o[-FRAME:][PREV_SLOT] = a
            wenv._history[-1][PREV_SLOT] = a
            wenv.prev_actions[:] = a
            if _tilt_gyro(wenv)[0] > TILT_FAULT:
                return {"ok": False, "phase": "stand" if i < 500 else "walk"}
            if i >= 550:
                vx.append(float((wenv._world_to_body_rot() @ wenv.data.qvel[:3])[0]))
        return {"ok": True, "walk_vx": float(np.mean(vx))}
    finally:
        wenv.close()


def _rate(xs) -> float:
    xs = [bool(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    succ = [r["success"] for r in rows]
    t = [r["t_handoff_s"] for r in rows if r["success"]]
    out = {"n": len(rows), "success": _rate(succ),
           "per_category": {c: _rate([r["success"] for r in rows if r["category"] == c])
                            for c in CATEGORIES if any(r["category"] == c for r in rows)},
           "median_t_handoff_s": float(np.median(t)) if t else float("nan"),
           "servo_sat_s_max": float(max(r["servo_sat_s"] for r in rows)),
           "peak_w_rads_p95": float(np.percentile([r["peak_w_rads"] for r in rows], 95)),
           "base_impact_n_p95": float(np.percentile([r["base_impact_n"] for r in rows], 95)),
           "inter_leg_n_p95": float(np.percentile([r["inter_leg_n"] for r in rows], 95))}
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], []).append(r["success"])
    full = [v for v in pairs.values() if len(v) == 2]
    out["mirror_disagree"] = float(np.mean([v[0] != v[1] for v in full])) if full else float("nan")
    ho = [r["handoff"]["ok"] for r in rows if r.get("handoff") is not None]
    out["handoff"] = _rate(ho) if ho else None
    return out


def verdict(summary: dict) -> list[tuple[str, bool]]:
    v, out = VERDICT, []
    for name, s in summary.items():
        lo = (v["nominal_success_min"] if name == "nominal" else v["weak_success_min"] if name == "weak"
              else v["other_success_min"])
        out.append((f"{name}: success {s['success']:.2f} >= {lo:.2f}", s["success"] >= lo))
        out.append((f"{name}: servo above {SAT_FRAC:.0%} stall <= {v['servo_sat_max_s']} s "
                    f"(max {s['servo_sat_s_max']:.2f} s)", s["servo_sat_s_max"] <= v["servo_sat_max_s"]))
        if s.get("handoff") is not None:
            out.append((f"{name}: hand-off to walking {s['handoff']:.2f} >= {v['handoff_min']:.2f}",
                        s["handoff"] >= v["handoff_min"]))
    if "nominal" in summary:
        s = summary["nominal"]
        worst = min(s["per_category"].items(), key=lambda kv: kv[1]) if s["per_category"] else ("-", float("nan"))
        out.append((f"nominal: every category >= {v['nominal_category_min']:.2f} (worst {worst[0]} {worst[1]:.2f})",
                    all(x >= v["nominal_category_min"] for x in s["per_category"].values())))
        out.append((f"nominal: median time-to-handoff <= {v['median_time_max_s']} s ({s['median_t_handoff_s']:.2f})",
                    s["median_t_handoff_s"] <= v["median_time_max_s"]))
        out.append((f"nominal: mirror-pair disagreement <= {v['mirror_disagree_max']:.2f} ({s['mirror_disagree']:.2f})",
                    s["mirror_disagree"] <= v["mirror_disagree_max"]))
    return out


def gate_starts(bank, per_category: int, seed: int) -> list[tuple[int, int]]:
    """(row, pair id): per category up to per_category ORIGINAL gate states plus each one's mirror row."""
    a, n_half = bank.arrays, len(bank) // 2
    rng = np.random.default_rng(seed)
    out = []
    for ci, _ in enumerate(CATEGORIES):
        orig = np.flatnonzero((a["split"] == 1) & ~a["mirrored"] & (a["category"] == ci))
        for i in rng.permutation(orig)[:per_category]:
            out += [(int(i), int(i)), (int(i) + n_half, int(i))]       # build_bank: mirror of row i is row i + n/2
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MuJoCo gate for an exported get-up bundle (format 7)")
    p.add_argument("--bundle", required=True, help="exported_getup/<name>")
    p.add_argument("--walk_bundle", default=None, help="exported/<name>: also test the hand-off to walking")
    p.add_argument("--bank", default=None, help="Default: the run config's env_config.getup.bank_path")
    p.add_argument("--per_category", type=int, default=8, help="Original gate states per category (each + mirror)")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--rate", type=float, default=0.5, help="Jetson rate limiter per step (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON report (default data_getup/gate/<bundle>.json)")
    args = p.parse_args(argv)

    policy = Policy(args.bundle)
    dc = policy.cfg
    if dc.get("policy_role") != "getup" or int(dc.get("format_version", 0)) != 7:
        print(f"[getup_gate] {args.bundle} is not a get-up bundle (policy_role {dc.get('policy_role')!r}, "
              f"format {dc.get('format_version')}); use src/mujoco/policy_gate.py for walking bundles", file=sys.stderr)
        return 1
    config = Config(_run_config_path(args.bundle))
    ec = config.train.env_config
    if int(config.train.history_size) != policy.hist:
        raise RuntimeError(f"run config history_size {config.train.history_size} != bundle {policy.hist}")
    g = dict(ec.getup or {})
    handoff, timeout_s = dict(dc["handoff"]), float(dc.get("timeout_s", g.get("episode_s", 10.0)))
    bank = load_bank(args.bank or g.get("bank_path", "data_getup/fall_bank.npz"), root_dir=_ROOT)
    starts = gate_starts(bank, args.per_category, args.seed)
    wpol, wec = None, None
    if args.walk_bundle:
        wpol = Policy(args.walk_bundle)
        if wpol.cfg.get("policy_role", "walk") == "getup":
            raise RuntimeError("--walk_bundle is a get-up bundle")
        wec = Config(_run_config_path(args.walk_bundle)).train.env_config
    print(f"[getup_gate] {args.bundle}: {len(starts)} gate starts ({args.per_category}/category + mirrors), "
          f"conditions {args.conditions}, handoff {handoff}, timeout {timeout_s} s"
          + (f", walk bundle {args.walk_bundle}" if wpol else ""))

    results, summary = {}, {}
    for cname in args.conditions:
        cond = CONDITIONS[cname]
        env = _make_env(ec, policy.hist, cond, args.seed)
        rows, t0 = [], time.time()
        try:
            for k, (row, pair) in enumerate(starts):
                r, env = run_getup(policy, env, bank.row(row), args.seed + k, cond, args.rate, handoff, timeout_s)
                r.update(category=CATEGORIES[int(bank.arrays["category"][row])], pair=pair, row=row)
                if wpol is not None and r["success"]:
                    r["handoff"] = run_handoff(env, wpol, wec, cond, args.seed + k, args.rate)
                rows.append(r)
        finally:
            env.close()
        results[cname] = rows
        summary[cname] = s = summarize(rows)
        cats = " ".join(f"{c}={v:.2f}" for c, v in s["per_category"].items())
        print(f"[getup_gate] {cname:8s} success {s['success']:.2f} ({cats}) mirror-disagree {s['mirror_disagree']:.2f} "
              f"t50 {s['median_t_handoff_s']:.2f}s sat {s['servo_sat_s_max']:.2f}s w95 {s['peak_w_rads_p95']:.1f} "
              f"impact95 {s['base_impact_n_p95']:.0f}N legs95 {s['inter_leg_n_p95']:.1f}N"
              + (f" handoff {s['handoff']:.2f}" if s.get("handoff") is not None else "")
              + f"  [{time.time() - t0:.0f}s]")

    checks = verdict(summary)
    print("\n[getup_gate] verdict:")
    for text, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
    passed = all(ok for _, ok in checks)
    print(f"[getup_gate] {'PASS' if passed else 'FAIL'}")
    out = args.out or os.path.join(_ROOT, "data_getup", "gate", os.path.basename(os.path.normpath(args.bundle)) + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"bundle": args.bundle, "walk_bundle": args.walk_bundle, "summary": summary,
                   "verdict": [{"check": t, "pass": ok} for t, ok in checks], "pass": passed,
                   "results": results, "thresholds": VERDICT}, fh, indent=2, default=float)
    print(f"[getup_gate] report: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
