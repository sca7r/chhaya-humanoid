"""SuperMimic training loop (Algorithm A.1): collect -> fit the world model -> improve the policy
by backpropagating imagined returns through the frozen world model, with a critic bootstrapping
beyond the horizon (Eqs. 1-3).

The env is anything with this interface (see ``isaac_adapter.ChhayaAdapter`` and the toy env in
``tests/test_supermimic.py``), all tensors on ``env.device``:

    n, device, state_dim, action_dim, ctx_dim, layout (world_model.Layout)
    reset()                      -> None
    state()                      -> (n, S)   world-model state, noise-free
    context()                    -> (n, C)   per-episode sim context xi
    policy_obs()                 -> (n, history * F)  what the actor sees in the simulator
    step(action (n, A))          -> reward (n,), terminated (n,) bool, truncated (n,) bool
"""
import math
import os
import time
from dataclasses import asdict

import torch
from torch import nn

from .config import SuperMimicConfig
from .replay import Replay
from .world_model import WorldModel, mlp


class Actor(nn.Module):
    """Deterministic policy on the normalised observation history; tanh-bounded actions."""

    def __init__(self, frame: int, history: int, action_dim: int, hidden, bound: float):
        super().__init__()
        self.net = mlp(frame * history, hidden, action_dim)
        with torch.no_grad():   # start at ~zero action: Chhaya's home pose, so the first data is of standing
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.bound, self.frame, self.history = bound, frame, history
        self.register_buffer("o_mean", torch.zeros(frame))
        self.register_buffer("o_std", torch.ones(frame))

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """hist (B, history, F) or (B, history * F)."""
        h = hist.reshape(hist.shape[0], self.history, self.frame)
        x = ((h - self.o_mean) / self.o_std).reshape(h.shape[0], -1)
        return torch.tanh(self.net(x)) * self.bound


class SuperMimic:
    def __init__(self, env, cfg: SuperMimicConfig, log_dir: str = None):
        torch.manual_seed(cfg.seed)
        self.env, self.cfg, self.dev = env, cfg, env.device
        lay = env.layout
        S, A, C, F = env.state_dim, env.action_dim, env.ctx_dim, len(lay.obs)
        self.obs_idx = torch.tensor(lay.obs, dtype=torch.long, device=self.dev)
        self.noise_std = torch.tensor(lay.obs_noise_std, dtype=torch.float32, device=self.dev)
        self.wm = WorldModel(S, A, C, lay, cfg.wm_hidden, cfg.head_hidden).to(self.dev)
        self.actor = Actor(F, lay.history, A, cfg.actor_hidden, lay.action_bound).to(self.dev)
        self.critic = mlp(S + C, cfg.critic_hidden, 1).to(self.dev)
        opt = lambda m: torch.optim.Adam(m.parameters(), lr=cfg.lr, betas=cfg.adam_betas)
        self.opt_wm, self.opt_actor, self.opt_critic = opt(self.wm), opt(self.actor), opt(self.critic)
        self.replay = Replay(env.n, cfg.buffer_steps, S, A, C, self.dev)
        self.it, self.env_steps, self._step_credit = 0, 0, 0.0
        self.log_dir = log_dir
        self.writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                pass
        self._ep_ret = torch.zeros(env.n, device=self.dev)
        self._ep_len = torch.zeros(env.n, device=self.dev)
        self._finished = []                             # (return, length) of finished episodes since last log

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def collect(self, steps: int, noise: float) -> None:
        env = self.env
        s = env.state()
        for _ in range(steps):
            c = env.context()
            a = self.actor(env.policy_obs())
            a = (a + noise * torch.randn_like(a)).clamp(-env.layout.action_bound, env.layout.action_bound)
            r, term, trunc = env.step(a)
            if hasattr(env, "executed_action"):      # the env may perturb actions itself (action-noise DR)
                a = env.executed_action()
            done = term | trunc
            self.replay.add(s, a, r, term, done, c)
            s_next = env.state()
            ok = ~done
            self.wm.update_stats(s[ok], s_next[ok], c[ok], r[ok])
            s = s_next
            self._ep_ret += r
            self._ep_len += 1
            if done.any():
                self._finished.append((self._ep_ret[done].clone(), self._ep_len[done].clone()))
                self._ep_ret[done] = 0.0
                self._ep_len[done] = 0.0
            self.env_steps += 1
        self._sync_actor_norm()

    @torch.no_grad()
    def _sync_actor_norm(self) -> None:
        """The actor sees noisy observation frames: its normaliser is the state's, plus the sensor noise."""
        self.actor.o_mean.copy_(self.wm.s_mean[self.obs_idx])
        self.actor.o_std.copy_(torch.sqrt(self.wm.s_std[self.obs_idx] ** 2 + self.noise_std ** 2).clamp(min=1e-3))

    # ------------------------------------------------------------------ world model
    def fit_world_model(self) -> dict:
        cfg, out = self.cfg, {}
        for _ in range(cfg.wm_batches_per_update):
            loss, out = self.wm.window_loss(*self.replay.windows(cfg.wm_batch, cfg.k))
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), cfg.grad_clip)
            self.opt_wm.step()
        return out

    # ------------------------------------------------------------------ policy and critic
    def imagine(self, s, c, hist, grad: bool = True):
        """H imagined steps on-policy through the frozen world model. Returns the objective J (Eq. 2)
        and the detached trajectory (states, rewards, termination probabilities) for the critic."""
        cfg = self.cfg
        states, rewards, pterm = [s], [], []
        J = torch.zeros(s.shape[0], device=self.dev)
        w, disc = torch.ones_like(J), 1.0
        for _ in range(cfg.horizon):
            a = self.actor(hist)
            s, r, logit = self.wm(s, a, c)
            p = torch.sigmoid(logit)
            J = J + disc * w * r
            w, disc = w * (1.0 - p), disc * cfg.gamma
            o = s[:, self.obs_idx]
            o = o + torch.randn_like(o) * self.noise_std
            hist = torch.cat([hist[:, 1:], o.unsqueeze(1)], dim=1)
            states.append(s); rewards.append(r); pterm.append(p)
        lo, hi = self.wm.value_bounds(cfg.gamma)
        V = self.critic(torch.cat([self.wm.norm_s(s), self.wm.norm_c(c)], -1)).squeeze(-1).clamp(lo, hi)
        J = J + disc * w * V
        traj = (torch.stack(states, 1).detach(), torch.stack(rewards, 1).detach(), torch.stack(pterm, 1).detach())
        return J, traj

    def improve_policy(self) -> dict:
        cfg = self.cfg
        s, c, hist = self.replay.seeds(cfg.imagined_batch, self.env.layout.history, self.obs_idx, self.noise_std)
        for p in self.wm.parameters():
            p.requires_grad_(False)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        J, (S_, R_, P_) = self.imagine(s, c, hist)
        loss = -J.mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.clamp_(-cfg.actor_elementwise_clip, cfg.actor_elementwise_clip)
        self.opt_actor.step()
        for p in self.wm.parameters():
            p.requires_grad_(True)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        # critic: GAE(lambda) targets over the imagined steps (Eq. 3), fixed while the critic is fit
        x = torch.cat([self.wm.norm_s(S_), self.wm.norm_c(c).unsqueeze(1).expand(-1, S_.shape[1], -1)], -1)
        with torch.no_grad():
            lo, hi = self.wm.value_bounds(cfg.gamma)
            V = self.critic(x).squeeze(-1).clamp(lo, hi)        # (B, H+1)
            R = V[:, -1]
            targets = []
            for t in reversed(range(cfg.horizon)):
                R = R_[:, t] + cfg.gamma * (1.0 - P_[:, t]) * ((1.0 - cfg.lam) * V[:, t + 1] + cfg.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1], 1).clamp(lo, hi)   # (B, H)
        xs, ys = x[:, :-1].reshape(-1, x.shape[-1]), targets.reshape(-1)
        closs = torch.zeros((), device=self.dev)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(xs.shape[0], device=self.dev)
            for mb in perm.chunk(cfg.critic_minibatches):
                closs = (self.critic(xs[mb]).squeeze(-1) - ys[mb]).square().mean()
                self.opt_critic.zero_grad(set_to_none=True)
                closs.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                self.opt_critic.step()
        return {"policy/J": J.mean().detach(), "policy/imagined_reward": R_.mean(),
                "policy/imagined_survival": (1.0 - P_).prod(1).mean(), "critic/loss": closs.detach()}

    # ------------------------------------------------------------------ loop
    def _set_lr(self) -> None:
        cfg = self.cfg
        f = min(self.it, cfg.lr_decay_iters) / max(cfg.lr_decay_iters, 1)
        scale = cfg.lr_final_frac + (1.0 - cfg.lr_final_frac) * 0.5 * (1.0 + math.cos(math.pi * f))
        for o in (self.opt_wm, self.opt_actor, self.opt_critic):
            for g in o.param_groups:
                g["lr"] = cfg.lr * scale

    def train(self, iterations: int = None, minutes: float = None, on_log=None) -> nn.Module:
        cfg = self.cfg
        iterations = cfg.iterations if iterations is None else iterations
        if self.replay.T == 0:
            self.env.reset()
            self.collect(max(cfg.warmup_env_steps, cfg.k + self.env.layout.history + 2), cfg.action_noise)
        t0, last = time.time(), time.time()
        while self.it < iterations and (minutes is None or time.time() - t0 < 60.0 * minutes):
            self._step_credit += cfg.env_steps_per_iter
            n = int(self._step_credit)
            self._step_credit -= n
            n = max(n, -(-cfg.min_transitions_per_iter // self.env.n))   # ceil: the data floor
            self.collect(n, cfg.action_noise)
            self._set_lr()
            logs = self.fit_world_model()
            logs.update(self.improve_policy())
            self.it += 1
            if self.it % cfg.log_every == 0:
                logs = {k: float(v) for k, v in logs.items()}
                if self._finished:
                    rets = torch.cat([f[0] for f in self._finished]); lens = torch.cat([f[1] for f in self._finished])
                    logs["env/episode_return"], logs["env/episode_length"] = float(rets.mean()), float(lens.mean())
                    self._finished = []
                now = time.time()
                logs["perf/iters_per_s"] = cfg.log_every / max(now - last, 1e-9)
                logs["perf/env_steps"] = float(self.env_steps * self.env.n)
                last = now
                if self.writer:
                    for k, v in logs.items():
                        self.writer.add_scalar(k, v, self.it)
                if on_log:
                    on_log(self.it, logs)
            if self.log_dir and self.it % cfg.save_every == 0:
                self.save(os.path.join(self.log_dir, f"supermimic_{self.it}.pt"))
        if self.log_dir:
            self.save(os.path.join(self.log_dir, "supermimic_last.pt"))
        return self.actor

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "world_model": self.wm.state_dict(),
                    "critic": self.critic.state_dict(), "config": asdict(self.cfg),
                    "layout": vars(self.env.layout), "iteration": self.it, "env_steps": self.env_steps,
                    "dims": {"state": self.env.state_dim, "action": self.env.action_dim, "ctx": self.env.ctx_dim}},
                   path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.dev)
        self.actor.load_state_dict(ck["actor"]); self.wm.load_state_dict(ck["world_model"])
        self.critic.load_state_dict(ck["critic"]); self.it = ck["iteration"]
