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
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval", default=None, help="checkpoint: report velocity tracking (policy vs standing still) instead of training")
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
def evaluate(ad, policy, steps=1000):
    """Mean velocity-tracking error over all robots, 20 s, no exploration noise."""
    e, m = ad.env, {"lin_err": 0.0, "yaw_err": 0.0, "cmd_speed": 0.0, "speed": 0.0}
    ad.reset()
    falls = 0
    for _ in range(steps):
        _, term, _ = ad.step(policy(ad.policy_obs()))
        c, v, w = e.commands, e.rd.root_lin_vel_b[:, :2], e.rd.root_ang_vel_b[:, 2]
        m["lin_err"] += float((c[:, :2] - v).norm(dim=1).mean()); m["yaw_err"] += float((c[:, 2] - w).abs().mean())
        m["cmd_speed"] += float(c[:, :2].norm(dim=1).mean()); m["speed"] += float(v.norm(dim=1).mean())
        falls += int(term.sum())
    return {**{k: round(x / steps, 4) for k, x in m.items()}, "falls_per_robot": round(falls / ad.n, 3)}


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
    smc = {"paper": SuperMimicConfig(), "laptop": laptop(), "tiny": tiny()}[args.preset]
    smc = replace(smc, seed=args.seed, model=args.model, **({"lr": args.lr} if args.lr else {}),
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
        sm = SuperMimic(ad, smc, log_dir=None if args.eval else log_dir)
        if args.eval:
            sm.load(args.eval)
            for who, pol in (("policy", sm.actor), ("standing", lambda o: torch.zeros(ad.n, ad.action_dim, device=ad.device))):
                print("[supermimic] eval", who, evaluate(ad, pol), flush=True)
            env.close()
            return

        def show(it, logs):
            keys = ("wm/state", "wm/reward", "wm/term", "policy/J", "policy/imagined_reward",
                    "policy/imagined_survival", "critic/loss", "env/episode_return", "env/episode_length",
                    "env/model_reward",
                    "perf/iters_per_s", "perf/env_steps")
            print(f"[supermimic] it {it} " + " ".join(f"{k.split('/')[1]}={logs[k]:.4g}" for k in keys if k in logs),
                  flush=True)
        sm.train(iterations=args.iterations, minutes=args.minutes, on_log=show)
        print("[supermimic] saved", os.path.join(log_dir, "supermimic_last.pt"))
        env.close()


if __name__ == "__main__":
    main()
