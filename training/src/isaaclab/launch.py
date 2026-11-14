"""Simulator start-up shared by train.py / play.py (Isaac Lab 3.0, Newton backend, no Isaac Sim).

Isaac Lab 2.x started Omniverse Kit with AppLauncher BEFORE anything could be imported. In 3.0 the
Newton backend is a normal Python library: build the env cfg first, then open the simulation with
`launch_simulation(cfg, args)` and create the env inside it.
"""
from __future__ import annotations

import argparse
import os


def add_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default=None,
                        help="cuda:0 | cpu. Default: cuda:0 if a CUDA GPU is present, else cpu (slow; for tests).")
    parser.add_argument("--viz", type=str, default="none",
                        help="Viewer: none (training) | newton | rerun | viser (see Isaac Lab 3.0 visualizers).")


def resolve_device(args) -> str:
    if args.device:
        return args.device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def launch(env_cfg, args):
    """Context manager: `with launch(env_cfg, args): env = BipedEnv(env_cfg)`."""
    os.environ.setdefault("HEADLESS", "1" if args.viz == "none" else "0")
    from isaaclab.app import launch_simulation
    device = resolve_device(args)
    env_cfg.sim.device = device
    return launch_simulation(env_cfg, argparse.Namespace(device=device, viz=args.viz, headless=args.viz == "none"))
