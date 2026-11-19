"""SuperMimic settings.

Defaults follow the paper's Table A.5 (one H100, 4096 envs) where our task allows; network sizes
are smaller because Chhaya's state is ~140 dims, not the G1 motion tracker's. ``laptop()`` is a
preset sized for a 6 GB GPU (RTX 3060 laptop). Paper: Liao et al., "SuperMimic: Scaling Humanoid
Whole-Body Control with One Differentiable World Model", CoRL 2026, arXiv:2610.28480.
"""
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class SuperMimicConfig:
    # "eth": dynamics-only world model, analytic reward and termination (eth.py); "v1": learned reward/termination heads
    model: str = "v1"                               # train_supermimic.py defaults to "eth"
    # exactly left-right symmetric actor: a(s) = 1/2 (f(s) + M_a f(M_s s)); needs env.mirror_tables()
    mirror: bool = False                            # train_supermimic.py turns it on (the eth runs limped)
    # --- world model (paper: MLP 2048x3, k=8, Adam 2e-4, 10 batches of 2048 windows per update) ---
    wm_hidden: Tuple[int, ...] = (1024, 1024, 1024)
    head_hidden: Tuple[int, ...] = (512, 512)       # reward and termination heads
    k: int = 8                                      # autoregressive window length for the fit
    wm_batch: int = 2048
    wm_batches_per_update: int = 10
    # --- imagination and policy (paper: H=32, 4096 rollouts, gamma 0.99, clip 1.0 then 0.1) ---
    horizon: int = 32
    imagined_batch: int = 4096
    actor_hidden: Tuple[int, ...] = (512, 256, 128)
    critic_hidden: Tuple[int, ...] = (256, 128)     # paper: (256, 128)
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    # --- optimisation (one learning rate for all three, cosine decay to 0.03x, then constant) ---
    lr: float = 2e-4
    adam_betas: Tuple[float, float] = (0.9, 0.95)   # paper: policy Adam beta (0.9, 0.95)
    grad_clip: float = 1.0
    actor_elementwise_clip: float = 0.1
    lr_decay_iters: int = 250_000
    lr_final_frac: float = 0.03
    iterations: int = 320_000
    # --- data collection (paper: action noise N(0, 0.1^2), 1000 replay steps per env, update/data 0.4) ---
    action_noise: float = 0.1
    buffer_steps: int = 1000
    env_steps_per_iter: float = 2.5                 # = 1 / (update-to-data ratio 0.4)
    # Floor on new transitions per iteration (all envs together). The paper collects ~10k per update
    # (4096 envs x 2.5); with few envs the policy is updated far more often per sample and overfits a
    # poorly trained model (seen on Chhaya with 4 CPU envs). 0 disables the floor.
    min_transitions_per_iter: int = 2048
    warmup_env_steps: int = 64                      # before the first update; also fixes the normalisers
    # --- bookkeeping ---
    log_every: int = 50
    save_every: int = 5000
    seed: int = 0


def laptop(cfg: SuperMimicConfig = SuperMimicConfig()) -> SuperMimicConfig:
    """Sized for a 6 GB GPU with ~1024 envs: replay ~0.3 GB, imagination activations ~1 GB."""
    return replace(cfg, wm_hidden=(512, 512, 512), head_hidden=(256, 256), wm_batch=1024,
                   imagined_batch=1024, buffer_steps=300)


def tiny(cfg: SuperMimicConfig = SuperMimicConfig()) -> SuperMimicConfig:
    """CPU smoke tests only. The higher learning rate makes up for running hundreds, not 320k, iterations."""
    return replace(cfg, wm_hidden=(256, 256), head_hidden=(128, 128), actor_hidden=(64, 64), critic_hidden=(64,),
                   wm_batch=128, wm_batches_per_update=2, imagined_batch=128, horizon=16, buffer_steps=200,
                   warmup_env_steps=32, lr=1e-3, log_every=10, save_every=10**9, min_transitions_per_iter=0)
