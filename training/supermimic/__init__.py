"""SuperMimic-style model-based training for Chhaya: one learned, differentiable world model of the
simulator, and a policy improved by backpropagating imagined returns through it.

Reimplemented from the paper (no reference code was available): Liao, Shao, Long, Zakka, Horowitz,
Abbeel, Sreenath, "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model", CoRL 2026, arXiv:2610.28480. Differences from the paper: docs/SUPERMIMIC.md.
"""
from .config import SuperMimicConfig, laptop, tiny
from .learner import Actor, SuperMimic
from .world_model import Layout, WorldModel

__all__ = ["SuperMimicConfig", "laptop", "tiny", "SuperMimic", "Actor", "Layout", "WorldModel"]
