
import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Chhaya in Isaac Lab 3.0 (Newton)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Chhaya",
                    help="Chhaya (walking) | ChhayaGetUp (fall recovery)")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--config_path", type=str, default="config/chhaya.yaml", help="Path to training config")
parser.add_argument("--steps", type=int, default=None, help="Override train.steps (short test runs).")

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
_training_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
for _p in (os.path.join(_training_root, "torch_rl_algorithms"), _project_root, _training_root):
    # torch_rl_algorithms uses bare imports (from models.networks import ...), so its own directory
    # must be on sys.path as well.
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from src.isaaclab.launch import add_launch_args, launch

add_launch_args(parser)
args_cli = parser.parse_args()

from torch_rl_algorithms.algorithms.sac.model import SAC
from torch_rl_algorithms.algorithms.ddpg.model import DDPG
from torch_rl_algorithms.algorithms.d4pg.model import D4PG
from torch_rl_algorithms.algorithms.mpo.model import MPO
from torch_rl_algorithms.algorithms.ppo.model import PPO
from config.schema import ModelType
from utils import Config
from utils.residual_home import freeze_residual_map
from src.isaaclab.common import BaseIsaacLabWrapper
from src.isaaclab.env_config import build_env_cfg, describe as describe_env_cfg
from src.isaaclab.tasks import resolve_task

BipedEnvCfg, BipedEnv = resolve_task(args_cli.task)

IsaacLabWrapper = BaseIsaacLabWrapper

def train():
    # Resolve Config Path from Checkpoint if applicable
    if args_cli.checkpoint and "--config_path" not in sys.argv:
        # User provided checkpoint but not config. Try to find config in checkpoint dir.
        ckpt_path = os.path.abspath(args_cli.checkpoint)
        ckpt_dir = os.path.dirname(ckpt_path)
        
        # Check in the checkpoint directory
        found_config = None
        # 1. Find first fide with .yaml or .yml extension
        for name in os.listdir(ckpt_dir):
            if name.endswith(".yaml") or name.endswith(".yml"):
                found_config = os.path.join(ckpt_dir, name)
                break
                    
        if found_config:
            print(f"[INFO] Auto-detected config from checkpoint: {found_config}")
            args_cli.config_path = found_config

    # Load config
    # Adjust path if running from different directory
    if not os.path.exists(args_cli.config_path):
        # Try relative to this file
        rel_path = os.path.join(os.path.dirname(__file__), "../../../", args_cli.config_path)
        if os.path.exists(rel_path):
            args_cli.config_path = rel_path
        else:
            print(f"Config file not found: {args_cli.config_path}")
            return

    config = Config(args_cli.config_path)
    # action_home_deg / action_scale_rad / spawn_height_m may say `auto` (derived from the gait table and the robot model):
    # freeze the numbers into this config NOW, before the env is built and before the config is saved to
    # checkpoints/<run>/config.yaml, so a later re-bake cannot change an existing run's map. No-op for explicit numbers.
    freeze_residual_map(config)
    # TF32 matmuls (train.tf32): the learner is ~90 ms/it of fp32 GEMM at batch 2048.
    torch.backends.cuda.matmul.allow_tf32 = bool(config.train.tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.train.tf32)

    # Initialize Isaac Lab Environment. All config.yaml -> BipedEnvCfg mapping lives in
    # src/isaaclab/env_config.py, shared with play.py, so the two cannot drift apart again.
    env_cfg = build_env_cfg(config, BipedEnvCfg, training=True, num_envs=args_cli.num_envs)
    print("[train] env contract:", describe_env_cfg(env_cfg))

    # Set seeds
    seed = args_cli.seed if args_cli.seed is not None else config.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Pass seed to config for environment
    env_cfg.seed = seed

    # Generate timestamp now so it's available for the video directory
    log_dir = config.train.log_dir
    model_name = config.train.model_name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"{model_name}_{timestamp}"
    if config.train.checkpoint_path:
        config.train.checkpoint_path = os.path.join(config.train.checkpoint_path, model_name)

    if args_cli.steps is not None:
        config.train.steps = args_cli.steps

    # Open the simulation (kit-less Newton) and run everything inside it.
    with launch(env_cfg, args_cli):
        _run(config, env_cfg, log_dir, model_name)


def _run(config, env_cfg, log_dir, model_name):
    env = BipedEnv(cfg=env_cfg, render_mode=None)

    # Wrap environment
    # With replay mirror augmentation on, make the stage-1 obs statistics mirror-symmetric (see
    # BaseIsaacLabWrapper). The wrapper reaches BipedEnv.mirror_obs through env.unwrapped, so the
    # --video RecordVideo wrapper is fine. play/export/probe paths never pass this (stats frozen).
    wrapped_env = IsaacLabWrapper(
        env, config,
        symmetric_obs_stats=bool(getattr(config.train, "symmetry_augmentation", False)),
    )

    # Initialize Model (SAC)
    device = torch.device(env.device)          # learner on the simulator's device
    print(f"Using device: {device}")

    use_history = config.train.use_history
    history_size = config.train.history_size
    
    # Select model class based on config
    model_type = config.train.model
    if model_type == ModelType.SAC:
        ModelClass = SAC
    elif model_type == ModelType.DDPG:
        ModelClass = DDPG
    elif model_type == ModelType.D4PG:
        ModelClass = D4PG
    elif model_type == ModelType.MPO:
        ModelClass = MPO
    elif model_type == ModelType.PPO:
        ModelClass = PPO
    else:
        # Fallback for string values if not using Enum directly or if config loaded as dict
        if str(model_type).lower() == "sac":
            ModelClass = SAC
        elif str(model_type).lower() == "ddpg":
            ModelClass = DDPG
        elif str(model_type).lower() == "d4pg":
            ModelClass = D4PG
        elif str(model_type).lower() == "mpo":
            ModelClass = MPO
        elif str(model_type).lower() == "ppo":
            ModelClass = PPO
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model = ModelClass(
        env=wrapped_env,
        device=device,
        config=config,
        use_history=use_history,
        history_size=history_size,
        model_path=args_cli.checkpoint # Resume if checkpoint provided
    )
    
    # Train
    model.train(
        log_dir=log_dir,
        log_name=model_name,
        steps=config.train.steps,
        save_steps=config.train.save_steps
    )
    
    # Close
    env.close()

if __name__ == "__main__":
    train()
