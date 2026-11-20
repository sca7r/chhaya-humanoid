"""Generate the get-up fall bank (utils/fall_bank.py): settled fallen states, captured the way the Jetson re-arms.

Hardware protocol (2026-11-01): the walking policy faults at 45 deg tilt and turns the servo torque OFF, the robot
falls limp, and the Jetson re-arms once it lies still (getup.settle: |gyro| < gyro_rads and every |qd| < joint_vel_rads
held for hold_s). This script reproduces that in the get-up env itself (same physics, BAM servo model, backlash, DR):

  drop  random orientation (uniform SO(3)) 0.30-0.40 m up, random joints (half uniform in the joint box, half
        home + N(0, 0.3 rad)), torque OFF from the start: every way of lying on the floor.
  push  standing at home, servos holding, kicked over (0.5-1.5 m/s + random spin); torque cut when the tilt passes
        45 deg, exactly the walking fault. The realistic walking fall.
  hold  like drop, but the servos hold the initial pose at a low P (4-12): rests that a stiffer-than-modelled
        unpowered gearbox would produce (the BAM model underestimates torque-off stiction).
Limp joints get their friction budget scaled x U(1, 3) for the same reason. Physics DR runs at full width.

A state is captured at the first step its settle criterion holds (and settle.min_s has passed, the base is not
moving: a free-falling robot has a quiet gyro and quiet joints); envs that never settle within settle.max_s are
dropped. The bank stores every state plus its exact mirror and a held-out gate split (build_bank).

    python src/isaaclab/make_fall_bank.py --config_path config/config_getup.yaml --headless [--num_envs 4096 --rounds 4]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate the get-up fall bank")
parser.add_argument("--config_path", type=str, default="config/config_getup.yaml")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--out", type=str, default=None, help="Default: the config's env_config.getup.bank_path")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gate_frac", type=float, default=0.1, help="Held-out fraction for the MuJoCo gate")
parser.add_argument("--mix", type=float, nargs=3, default=(0.5, 0.35, 0.15), metavar=("DROP", "PUSH", "HOLD"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

from src.isaaclab.env_config import build_env_cfg  # noqa: E402
from src.isaaclab.tasks import resolve_task  # noqa: E402
from utils import Config  # noqa: E402
from utils.fall_bank import CATEGORIES, SOURCES, build_bank, categorize, projected_gravity, remove_yaw, save_bank  # noqa: E402
from utils.residual_home import freeze_residual_map  # noqa: E402

TILT_CUT_RAD = 0.784       # the walking fault (biped_env._tilt_exceeded, deploy_config tilt_fault_rad)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_project_root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    config = Config(args_cli.config_path)
    freeze_residual_map(config)
    cfg_cls, env_cls = resolve_task("ChhayaGetUp")
    env_cfg = build_env_cfg(config, cfg_cls, training=True, num_envs=args_cli.num_envs, pushes=False)
    g = dict(env_cfg.getup)
    settle = dict(g["settle"])
    out = args_cli.out or g["bank_path"]
    # Every reset = the walking stand (no bank yet); no assist force; no episode time-outs inside a round; DR at full
    # width (curriculum off), no exploration noise on the (mostly ignored) actions.
    env_cfg.getup = {**g, "start_mix": {"bank": 0.0, "standing": 1.0}, "h_floor_m": 0.05,
                     "assist": {**g["assist"], "enabled": False}, "terminate_on_success": False}
    env_cfg.episode_length_s = 1.0e4
    env_cfg.curriculum_enabled = False
    env_cfg.action_noise_model = None
    env_cfg.seed = args_cli.seed
    env = env_cls(cfg=env_cfg)
    dev, n, dt = env.device, env.num_envs, env.step_dt
    acts = env.bam_actuators
    if not acts:
        raise RuntimeError("the fall bank needs the BAM servo model (torque off / low-P hold)")
    kern = acts[0].servo.kernel
    lo, hi = env.joint_limits_min, env.joint_limits_max
    home = env.default_joint_pos
    hold_steps = max(1, int(math.ceil(float(settle["hold_s"]) / dt - 1e-6)))
    max_steps = int(math.ceil(float(settle["max_s"]) / dt))
    p = torch.tensor(args_cli.mix, dtype=torch.float32, device=dev)
    p = p / p.sum()
    print(f"[fall-bank] {n} envs x {args_cli.rounds} rounds, mix drop/push/hold {p.tolist()}, settle {settle}, "
          f"<= {max_steps} steps per round")

    keep = {k: [] for k in ("root_z", "root_quat", "joint_pos", "joint_vel", "root_lin_vel_b", "root_ang_vel_b",
                            "source", "settle_time_s")}
    n_lost = 0
    for rnd in range(args_cli.rounds):
        env.reset()
        ids = torch.arange(n, device=dev)
        src = torch.multinomial(p, n, replacement=True)                        # 0 drop, 1 push, 2 hold
        limp = src != 1
        # ---- initial states
        q = torch.where(torch.rand(n, 1, device=dev) < 0.5,
                        lo + (hi - lo) * torch.rand(n, env.num_joints, device=dev),
                        torch.clamp(home + 0.3 * torch.randn(n, env.num_joints, device=dev), lo, hi))
        quat = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)    # uniform SO(3)
        z = 0.30 + 0.10 * torch.rand(n, device=dev)
        ang = (torch.rand(n, 3, device=dev) * 2.0 - 1.0)
        li = ids[limp]
        if len(li):
            env.write_start_states(li, z[li], quat[li], q[li], torch.zeros_like(q[li]),
                                   torch.zeros(len(li), 3, device=dev), ang[li],
                                   yaw=torch.zeros(len(li), device=dev))
        pi = ids[~limp]
        if len(pi):                                    # pushed over from the walking stand: a world-frame kick
            heading = 2.0 * math.pi * torch.rand(len(pi), device=dev)
            speed = 0.5 + torch.rand(len(pi), device=dev)
            vel = torch.zeros(len(pi), 6, device=dev)
            vel[:, 0], vel[:, 1] = speed * torch.cos(heading), speed * torch.sin(heading)
            vel[:, 3:5] = (torch.rand(len(pi), 2, device=dev) * 2.0 - 1.0) * 2.0
            env.robot.write_root_velocity_to_sim(vel, env_ids=pi)
        # ---- actuation
        torque_on = (src != 0).float()                 # drop: off from the start; push: on until the tilt cut
        for act in acts:
            act.torque_enable[:] = torque_on.unsqueeze(-1)
        kp0, fs0 = kern.kp.clone(), kern.friction_scale.clone()
        hold = src == 2
        kern.kp[hold] = (4.0 + 8.0 * torch.rand(int(hold.sum()), 1, device=dev)).expand(-1, kern.num_joints)
        kern.friction_scale[limp] = fs0[limp] * (1.0 + 2.0 * torch.rand(int(limp.sum()), 1, device=dev))
        a_hold = env.targets_to_actions(env.filtered_targets.clone())     # hold envs: target = the initial pose
        actions = torch.where(hold.unsqueeze(-1), a_hold, torch.zeros_like(a_hold))

        cut = src != 1                                 # push envs are "cut" once they pass the fault tilt
        still = torch.zeros(n, dtype=torch.long, device=dev)
        captured = torch.zeros(n, dtype=torch.bool, device=dev)
        broken = torch.zeros(n, dtype=torch.bool, device=dev)
        for step in range(max_steps):
            _, _, term, trunc, _ = env.step(actions)
            broken |= term | trunc
            d = env.robot.data
            tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
            newly_cut = ~cut & (tilt > TILT_CUT_RAD)
            if newly_cut.any():
                for act in acts:
                    act.torque_enable[newly_cut] = 0.0
                cut |= newly_cut
            calm = ((torch.norm(d.root_ang_vel_b, dim=-1) < float(settle["gyro_rads"]))
                    & (d.joint_vel.abs().max(dim=-1)[0] < float(settle["joint_vel_rads"]))
                    & (torch.norm(d.root_lin_vel_w, dim=-1) < float(settle["lin_vel_ms"]))     # not in free fall
                    & cut & ((step + 1) * dt >= float(settle["min_s"])))
            still = torch.where(calm, still + 1, torch.zeros_like(still))
            cap = (still >= hold_steps) & ~captured & ~broken
            if cap.any():
                c = cap.nonzero(as_tuple=False).squeeze(-1)
                body_z = d.body_pos_w[c, :, 2] - env.scene.env_origins[c, 2:3]
                ok = (body_z.min(dim=-1)[0] > -0.01) & torch.isfinite(d.joint_pos[c]).all(dim=-1)
                c = c[ok]
                keep["root_z"].append((d.root_pos_w[c, 2] - env.scene.env_origins[c, 2]).cpu())
                keep["root_quat"].append(d.root_quat_w[c].cpu())
                keep["joint_pos"].append(d.joint_pos[c].cpu())
                keep["joint_vel"].append(d.joint_vel[c].cpu())
                keep["root_lin_vel_b"].append(d.root_lin_vel_b[c].cpu())
                keep["root_ang_vel_b"].append(d.root_ang_vel_b[c].cpu())
                keep["source"].append(src[c].cpu())
                keep["settle_time_s"].append(torch.full((len(c),), (step + 1) * dt))
                captured |= cap
            if bool(captured.all()):
                break
        n_round = int(captured.sum().item())
        n_lost += n - n_round
        print(f"[fall-bank] round {rnd + 1}/{args_cli.rounds}: captured {n_round}/{n} "
              f"(broken {int(broken.sum().item())}, never settled {n - n_round - int((broken & ~captured).sum().item())})")
        kern.kp.copy_(kp0)
        kern.friction_scale.copy_(fs0)
        for act in acts:
            act.torque_enable[:] = 1.0

    cat = lambda k: torch.cat(keep[k]).numpy().astype(np.float64)
    quat = remove_yaw(cat("root_quat"))
    states = {"root_z": cat("root_z"), "root_quat": quat, "joint_pos": cat("joint_pos"), "joint_vel": cat("joint_vel"),
              "root_lin_vel_b": cat("root_lin_vel_b"), "root_ang_vel_b": cat("root_ang_vel_b"),
              "category": categorize(projected_gravity(quat)), "source": cat("source").astype(np.int8)}
    settle_t = cat("settle_time_s")
    meta = {"joint_names": list(env.robot.joint_names), "settle": settle, "tilt_cut_rad": TILT_CUT_RAD,
            "generator": {"config": os.path.relpath(os.path.abspath(args_cli.config_path), _project_root),
                          "num_envs": n, "rounds": args_cli.rounds, "mix_drop_push_hold": list(args_cli.mix),
                          "seed": args_cli.seed, "lost": n_lost,
                          "settle_time_s_median": float(np.median(settle_t))},
            "joint_limits_rad": [[float(a), float(b)] for a, b in zip(lo.tolist(), hi.tolist())],
            "home_rad": home.tolist(), "spawn_height_m": float(env.h_stand),
            "created_utc": datetime.now(timezone.utc).isoformat(), "git_sha": _git_sha()}
    bank = build_bank(states, meta, gate_frac=args_cli.gate_frac, seed=args_cli.seed)
    path = out if os.path.isabs(out) else os.path.join(_project_root, out)
    save_bank(bank, path)
    print(f"[fall-bank] wrote {path}: {len(bank)} states (incl. mirrors), h_floor {bank.h_floor_m:.4f} m, "
          f"train {len(bank.select('train'))} / gate {len(bank.select('gate'))}")
    print(f"[fall-bank] categories {bank.histogram()}")
    print(f"[fall-bank] sources {dict(zip(SOURCES, np.bincount(bank.arrays['source'], minlength=len(SOURCES)).tolist()))}")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        sys.stdout.flush()
        os._exit(rc)          # Isaac's app.close() hangs
