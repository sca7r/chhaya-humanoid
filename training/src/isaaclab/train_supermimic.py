"""Train Chhaya with SuperMimic (learned world model + backpropagation through imagined rollouts)
on Isaac Lab 3.0 / Newton. The SAC trainer (train.py) is unchanged and remains the default.

    python src/isaaclab/train_supermimic.py --config_path config/chhaya.yaml --num_envs 1024 --preset laptop --minutes 120
    python src/isaaclab/train_supermimic.py --num_envs 4096 --preset paper              # 24 GB+ GPU
    python src/isaaclab/train_supermimic.py --num_envs 4 --preset tiny --iterations 30  # CPU smoke test

Follows the paper: flat ground and no random pushes (the deterministic world model cannot predict
a push; the paper drops them too). docs/SUPERMIMIC.md explains every difference.
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (os.path.join(ROOT, "torch_rl_algorithms"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402

from src.isaaclab.launch import add_launch_args, launch  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--config_path", default=os.path.join(ROOT, "config/chhaya.yaml"))
ap.add_argument("--task", default="BipedRobotV2")
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--preset", choices=("paper", "laptop", "tiny"), default="laptop")
ap.add_argument("--model", choices=("eth", "v1"), default="eth",
                help="eth: analytic reward/termination, dynamics-only world model; v1: learned reward head")
ap.add_argument("--iterations", type=int, default=None)
ap.add_argument("--minutes", type=float, default=None, help="wall-clock budget (the paper: 3 h on one H100)")
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--min_transitions", type=int, default=None,
                help="floor on new transitions per iteration (default 2048; the tiny preset has none)")
ap.add_argument("--rough", action="store_true", help="keep the rough terrain mix (the paper trains on flat ground)")
ap.add_argument("--pushes", action="store_true", help="keep random pushes (the paper drops them)")
ap.add_argument("--save_every", type=int, default=None, help="checkpoint every N iterations")
ap.add_argument("--no_mirror", action="store_true", help="plain actor (no exact left-right symmetry)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval", default=None, help="checkpoint: report velocity tracking (policy vs standing still) instead of training")
ap.add_argument("--record", default=None, help="with --eval: save robot 0's motion (.npz) for src/mujoco/replay_video.py")
ap.add_argument("--record_n", type=int, default=1, help="with --record: how many robots (a crowd shot when > 1)")
ap.add_argument("--command", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"), help="with --eval: fixed command")
ap.add_argument("--payload", type=float, default=None,
                help="with --eval: fixed extra torso mass [kg] instead of the random payload (mass tolerance check)")
ap.add_argument("--resume", default=None, help="checkpoint to continue training from (fine-tuning)")
ap.add_argument("--log_dir", default=os.path.join(ROOT, "runs"))
add_launch_args(ap)
args = ap.parse_args()

from utils import Config  # noqa: E402
from utils.residual_home import freeze_residual_map  # noqa: E402
from src.isaaclab.env_config import build_env_cfg, describe  # noqa: E402
from src.isaaclab.tasks import resolve_task  # noqa: E402
from supermimic import SuperMimic, SuperMimicConfig, laptop, tiny  # noqa: E402
from supermimic.isaac_adapter import ChhayaAdapter  # noqa: E402


@torch.no_grad()
def evaluate(ad, policy, steps=999, mdp=None, record=None):
    """Mean velocity-tracking error over all robots, 20 s, no exploration noise."""
    e, m = ad.env, {"lin_err": 0.0, "yaw_err": 0.0, "cmd_speed": 0.0, "speed": 0.0}
    ad.reset()
    falls, mt, rec = 0, {}, []
    tq_sq, tq_peak, a_sum = 0.0, 0.0, 0.0                      # joint torque: mean square and 99th-percentile peak over robots
    for _ in range(steps):
        s0 = ad.state()
        a = policy(ad.policy_obs())
        _, term, _ = ad.step(a)
        a_sum = a_sum + ad.executed_action()
        t = e.rd.applied_torque.abs()
        tq_sq = tq_sq + (t ** 2).mean(0) / steps
        tq_peak = torch.maximum(torch.as_tensor(tq_peak, device=t.device), t.quantile(0.99, dim=0))
        if record:
            d = e.rd
            rec.append(torch.cat([d.root_pos_w[:args.record_n] - e.scene.env_origins[:args.record_n],   # each from its own spawn
                                  d.root_quat_w[:args.record_n], d.joint_pos[:args.record_n],
                                  e.commands[:args.record_n]], -1).cpu())
        if mdp is not None:
            for k, x in mdp.reward(s0, ad.executed_action(), ad.state(), terms=True).items():
                mt[k] = mt.get(k, 0.0) + float(x.mean()) / steps
        c, v, w = e.commands, e.rd.root_lin_vel_b[:, :2], e.rd.root_ang_vel_b[:, 2]
        m["lin_err"] += float((c[:, :2] - v).norm(dim=1).mean()); m["yaw_err"] += float((c[:, 2] - w).abs().mean())
        m["cmd_speed"] += float(c[:, :2].norm(dim=1).mean()); m["speed"] += float(v.norm(dim=1).mean())
        falls += int(term.sum())
    if record:
        import numpy as np
        np.savez(record, traj=torch.stack(rec).numpy(), joint_names=np.array(e.robot.joint_names), dt=e.step_dt)
        print("[supermimic] recorded", record)
    # left-right asymmetry: each robot's mean leg action against its mirror image (0 = symmetric gait)
    op, os_, ap, as_ = ad.mirror_tables()
    am = a_sum / steps
    asym = float((am - am[:, ap] * as_)[:, :12].abs().max(1).values.mean())
    names, stall = list(e.robot.joint_names), 2.94           # STS3215 at 12 V
    top = sorted(range(len(names)), key=lambda i: -float(tq_peak[i]))[:5]
    print("[supermimic] eval torque (N m: rms / p99 peak / peak % of stall)",
          {names[i]: (round(float(tq_sq[i].sqrt()), 2), round(float(tq_peak[i]), 2), f"{100 * float(tq_peak[i]) / stall:.0f}%")
           for i in top})
    env_terms = {k: round(float(v.mean()) / (steps * e.step_dt), 4) for k, v in e.episode_sums.items()}
    print("[supermimic] eval model reward terms per step", {k: round(x, 5) for k, x in mt.items()})
    print("[supermimic] eval env reward terms per second (since last reset)", env_terms)
    return {**{k: round(x / steps, 4) for k, x in m.items()}, "falls_per_robot": round(falls / ad.n, 3),
            "leg_asym": round(asym, 3)}


def main():
    EnvCfg, Env = resolve_task(args.task)
    config = Config(args.config_path)
    freeze_residual_map(config)
    cfg = build_env_cfg(config, EnvCfg, training=True, num_envs=args.num_envs)
    cfg.seed = args.seed
    if not args.rough:
        from envs.isaaclab.biped_env_cfg import FLAT_TERRAIN_CFG
        cfg.use_rough_terrain = False
        cfg.terrain.terrain_generator = FLAT_TERRAIN_CFG
        cfg.scene.contact_forces.filter_shape_prim_expr = ["/World/ground/.*"]
    if not args.pushes:
        cfg.events.pop("push_robot", None)
    if args.payload is not None:
        cfg.curriculum_dr_events = [e for e in cfg.curriculum_dr_events if e != "randomize_payload"]
        cfg.events["randomize_payload"].params["mass_distribution_params"] = (args.payload, args.payload)
    smc = {"paper": SuperMimicConfig(), "laptop": laptop(), "tiny": tiny()}[args.preset]
    mirror = not args.no_mirror
    if args.eval:                                             # evaluate a checkpoint with the actor it was trained with
        mirror = bool(torch.load(args.eval, map_location="cpu")["config"].get("mirror", False))
    smc = replace(smc, seed=args.seed, model=args.model, mirror=mirror, **({"lr": args.lr} if args.lr else {}),
                  **({"save_every": args.save_every} if args.save_every else {}),
                  **({"min_transitions_per_iter": args.min_transitions} if args.min_transitions is not None else {}))
    name = f"{config.train.model_name.replace('_SAC', '')}_SuperMimic_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join(args.log_dir, name)
    print("[supermimic] env contract:", describe(cfg))
    with launch(cfg, args):
        env = Env(cfg=cfg)
        ad = ChhayaAdapter(env)
        print(f"[supermimic] state {ad.state_dim}, action {ad.action_dim}, context {ad.ctx_dim}, "
              f"observation frame {len(ad.layout.obs)} x {ad.layout.history}, envs {ad.n}, device {ad.device}")
        print("[supermimic] layout check (should be ~0):", ad.check_layout())
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "run.json"), "w") as f:
            json.dump({"supermimic": asdict(smc), "config_path": args.config_path, "num_envs": args.num_envs,
                       "flat": not args.rough, "pushes": args.pushes}, f, indent=2)
        if smc.mirror:
            from models.equivariant import check_involution
            op, os_, ap, as_ = (t.cpu() for t in ad.mirror_tables())
            check_involution(op, os_, "observation mirror"); check_involution(ap, as_, "action mirror")
            print("[supermimic] mirror-symmetric actor")
        sm = SuperMimic(ad, smc, log_dir=None if args.eval else log_dir)
        if args.resume:
            sm.load(args.resume)
            print(f"[supermimic] resumed from {args.resume} at iteration {sm.it}")
            if args.iterations:
                args.iterations += sm.it                      # --iterations counts the new ones
        if args.eval:
            sm.load(args.eval)
            if args.command:
                env.command_generator.set_manual(tuple(args.command))
            for who, pol in (("policy", sm.actor), ("standing", lambda o: torch.zeros(ad.n, ad.action_dim, device=ad.device))):
                rec = args.record if who == "policy" else None
                print("[supermimic] eval", who, evaluate(ad, pol, mdp=getattr(sm.wm, "mdp", None), record=rec), flush=True)
            env.close()
            return

        def show(it, logs):
            keys = ("wm/state", "wm/reward", "wm/term", "policy/J", "policy/imagined_reward",
                    "policy/imagined_survival", "critic/loss", "env/episode_return", "env/episode_length",
                    "env/model_reward", "perf/s_collect", "perf/s_wm", "perf/s_policy",
                    "perf/iters_per_s", "perf/env_steps")
            print(f"[supermimic] it {it} " + " ".join(f"{k.split('/')[1]}={logs[k]:.4g}" for k in keys if k in logs),
                  flush=True)
        sm.train(iterations=args.iterations, minutes=args.minutes, on_log=show)
        print("[supermimic] saved", os.path.join(log_dir, "supermimic_last.pt"))
        env.close()


if __name__ == "__main__":
    main()
