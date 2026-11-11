"""--task name -> (env cfg class, env class, default checkpoint root). Shared by train.py and play.py.

Imports are lazy (module:attr strings) because the env modules import Isaac Lab, which only works after the
AppLauncher has started the simulator.
"""
from __future__ import annotations

import importlib

# name: (cfg "module:Class", env "module:Class", default checkpoints root for play's auto-select)
TASKS: dict[str, tuple[str, str, str]] = {
    "BipedRobot": ("envs.isaaclab.biped_env_cfg:BipedEnvCfg", "envs.isaaclab.biped_env:BipedEnv", "checkpoints"),
    "BipedRobotV2": ("envs.isaaclab.biped_env_cfg:BipedRobotV2EnvCfg", "envs.isaaclab.biped_env:BipedEnv",
                     "checkpoints"),
    # Get-up / fall recovery (config/config_getup.yaml). Its outputs live in runs_getup/ and checkpoints_getup/.
    "BipedRobotV2GetUp": ("envs.isaaclab.getup_env_cfg:BipedRobotV2GetUpEnvCfg",
                          "envs.isaaclab.getup_env:BipedGetUpEnv", "checkpoints_getup"),
}


def _load(spec: str):
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def resolve_task(name: str):
    """(env cfg class, env class) for a --task name."""
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name!r}; expected one of {sorted(TASKS)}")
    cfg_spec, env_spec, _ = TASKS[name]
    return _load(cfg_spec), _load(env_spec)


def checkpoint_root(name: str) -> str:
    """Folder (relative to BipedRobot/) that play.py searches for the newest checkpoint of this task."""
    return TASKS[name][2] if name in TASKS else "checkpoints"
