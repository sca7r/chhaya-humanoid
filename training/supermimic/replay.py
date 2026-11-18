"""Fixed-size replay of simulator transitions, one ring per env, on the learner's device.

Step tau stores the state s_tau seen before acting, the executed (noisy) action, the reward and
termination of that transition, whether the episode ended there (terminated or timed out), and the
episode's sim context. After an episode end the next stored state is the new episode's first state.
"""
import torch


class Replay:
    def __init__(self, n_envs: int, steps: int, S: int, A: int, C: int, device):
        self.n, self.L, self.dev = n_envs, steps, device
        self.s = torch.zeros(steps, n_envs, S, device=device)
        self.a = torch.zeros(steps, n_envs, A, device=device)
        self.c = torch.zeros(steps, n_envs, C, device=device)
        self.r = torch.zeros(steps, n_envs, device=device)
        self.term = torch.zeros(steps, n_envs, device=device)
        self.done = torch.zeros(steps, n_envs, device=device)
        self.T = 0                                   # steps written so far

    def add(self, s, a, r, term, done, c) -> None:
        i = self.T % self.L
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.term[i], self.done[i], self.c[i] = term.float(), done.float(), c
        self.T += 1

    def _oldest(self) -> int:
        return self.T - min(self.T, self.L)

    def _pick(self, B: int, lo: int, hi: int):
        """B random (step, env) pairs with absolute step in [lo, hi]."""
        if hi < lo:
            raise RuntimeError(f"replay holds too few steps ({self.T}) for this request")
        tau = torch.randint(lo, hi + 1, (B,), device=self.dev)
        env = torch.randint(0, self.n, (B,), device=self.dev)
        return tau, env

    def windows(self, B: int, k: int):
        """k-step windows: s (B, k+1, S), a/r/term/done (B, k), c (B, C) (context of the first step)."""
        tau, env = self._pick(B, self._oldest(), self.T - 1 - k)
        steps = (tau.unsqueeze(1) + torch.arange(k + 1, device=self.dev)) % self.L
        e = env.unsqueeze(1)
        return (self.s[steps, e], self.a[steps[:, :k], e], self.r[steps[:, :k], e],
                self.term[steps[:, :k], e], self.done[steps[:, :k], e], self.c[steps[:, 0], env])

    def seeds(self, B: int, history: int, obs_idx: torch.Tensor, noise_std: torch.Tensor):
        """Start states for imagination with their policy observation history, rebuilt from the stored
        states: frame j = obs part of s_(tau-j) plus fresh sensor noise, or zeros before the episode
        began (the env zero-fills its history at reset). Returns s (B, S), c (B, C), hist (B, H, F)."""
        tau, env = self._pick(B, self._oldest() + history - 1, self.T - 1)
        frames, alive = [], torch.ones(B, device=self.dev)
        for j in range(history):
            t = (tau - j) % self.L
            if j > 0:   # an episode end at step tau-j means s_(tau-j+1) started a new episode
                alive = alive * (1.0 - self.done[t, env])
            o = self.s[t, env][:, obs_idx]
            frames.append((o + torch.randn_like(o) * noise_std) * alive.unsqueeze(-1))
        hist = torch.stack(frames[::-1], dim=1)
        t0 = tau % self.L
        return self.s[t0, env], self.c[t0, env], hist

    def transitions(self, max_n: int = 200_000):
        """Valid (s, s_next, c, r) samples for the normalisers."""
        lo, hi = self._oldest(), self.T - 2
        steps = torch.arange(lo, hi + 1, device=self.dev) % self.L
        s, s2 = self.s[steps], self.s[(steps + 1) % self.L]
        ok = self.done[steps] < 0.5
        s, s2, c, r = s[ok], s2[ok], self.c[steps][ok], self.r[steps][ok]
        if s.shape[0] > max_n:
            i = torch.randperm(s.shape[0], device=self.dev)[:max_n]
            s, s2, c, r = s[i], s2[i], c[i], r[i]
        return s, s2, c, r
