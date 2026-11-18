"""The differentiable world model: s' = f(s, a, xi), r = r(s, a, s', xi), p(term) = d(s', xi).

Deterministic, conditioned on the per-episode sim context xi (domain-randomisation values), as in
SuperMimic section 3.2. The dynamics net predicts the normalised state change. Slots whose next value
is known exactly are written analytically after the prediction (``Layout``): the previous-action
slots take the action just applied, shifted action slots take the old previous action, and held
slots (the velocity command) keep their value. Those slots then carry exact gradients to the action.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import nn


def mlp(inp: int, hidden, out: int, act=nn.ELU) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


@dataclass
class Layout:
    """Where things sit in the world-model state vector (all index lists into the state)."""
    obs: List[int]                                    # the noise-free policy observation frame, in order
    obs_noise_std: List[float]                        # sensor noise added to that frame in imagination
    action_dst: List[int] = field(default_factory=list)   # state slots that become clamp(action)[action_src]
    action_src: List[int] = field(default_factory=list)
    shift_dst: List[int] = field(default_factory=list)    # state slots that become old_state[shift_src]
    shift_src: List[int] = field(default_factory=list)
    hold: List[int] = field(default_factory=list)         # state slots that keep their value (the command)
    history: int = 1                                  # policy observation frames
    action_bound: float = 1.0


class WorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, ctx_dim: int, layout: Layout,
                 hidden: Tuple[int, ...], head_hidden: Tuple[int, ...]):
        super().__init__()
        S, A, C = state_dim, action_dim, ctx_dim
        self.S, self.A, self.C, self.layout = S, A, C, layout
        self.dyn = mlp(S + A + C, hidden, S)
        self.rew = mlp(2 * S + A + C, head_hidden, 1)
        self.term = mlp(S + C, head_hidden, 1)
        for name, n in (("s_mean", S), ("s_std", S), ("d_std", S), ("c_mean", C), ("c_std", C)):
            self.register_buffer(name, torch.zeros(n) if "mean" in name else torch.ones(n))
        self.register_buffer("r_mean", torch.zeros(()))
        self.register_buffer("r_std", torch.ones(()))
        # running statistics behind the normalisers (Welford), and the range of rewards seen in the simulator
        for name, n in (("_s", S), ("_d", S), ("_c", C), ("_r", 1)):
            self.register_buffer(name + "_n", torch.zeros(()))
            self.register_buffer(name + "_mu", torch.zeros(n))
            self.register_buffer(name + "_m2", torch.zeros(n))
        self.register_buffer("r_lo", torch.zeros(()))
        self.register_buffer("r_hi", torch.zeros(()))
        mask = torch.zeros(S, dtype=torch.bool)
        for idx in (layout.action_dst, layout.shift_dst, layout.hold):
            mask[list(idx)] = True
        self.register_buffer("analytic", mask)
        self.register_buffer("i_adst", torch.tensor(layout.action_dst, dtype=torch.long))
        self.register_buffer("i_asrc", torch.tensor(layout.action_src, dtype=torch.long))
        self.register_buffer("i_sdst", torch.tensor(layout.shift_dst, dtype=torch.long))
        self.register_buffer("i_ssrc", torch.tensor(layout.shift_src, dtype=torch.long))
        self.register_buffer("i_hold", torch.tensor(layout.hold, dtype=torch.long))

    @torch.no_grad()
    def update_stats(self, s: torch.Tensor, s_next: torch.Tensor, c: torch.Tensor, r: torch.Tensor,
                     window: float = 2e6) -> None:
        """Fold a batch of valid transitions (s -> s_next) into the running normalisers. The sample count
        is capped at ``window``, so the statistics keep following the data as the policy changes."""
        if s.shape[0] == 0:
            return
        first = bool(self._r_n == 0)
        for key, x in (("_s", s), ("_d", s_next - s), ("_c", c), ("_r", r.reshape(-1, 1))):
            n0, mu0, m20 = getattr(self, key + "_n"), getattr(self, key + "_mu"), getattr(self, key + "_m2")
            nb = float(x.shape[0])
            mub, m2b = x.mean(0), ((x - x.mean(0)) ** 2).sum(0)
            n = n0 + nb
            delta = mub - mu0
            mu = mu0 + delta * nb / n
            m2 = m20 + m2b + delta ** 2 * n0 * nb / n
            if n > window:                          # keep the variance, shrink the weight of the past
                m2, n = m2 * (window / n), torch.full_like(n, window)
            n0.copy_(n); mu0.copy_(mu); m20.copy_(m2)

        def std(key):
            sd = torch.sqrt(getattr(self, key + "_m2") / getattr(self, key + "_n").clamp(min=1))
            return torch.where(sd < 1e-6, torch.ones_like(sd), sd.clamp(min=1e-4))   # constant dims: leave as is
        self.s_mean.copy_(self._s_mu); self.s_std.copy_(std("_s")); self.d_std.copy_(std("_d"))
        self.c_mean.copy_(self._c_mu); self.c_std.copy_(std("_c"))
        self.r_mean.copy_(self._r_mu[0]); self.r_std.copy_(std("_r")[0])
        lo, hi = r.min(), r.max()
        self.r_lo.copy_(lo if first else torch.minimum(self.r_lo, lo))
        self.r_hi.copy_(hi if first else torch.maximum(self.r_hi, hi))

    def value_bounds(self, gamma: float):
        """The range any discounted return can take, given the rewards seen so far."""
        return float(torch.clamp(self.r_lo, max=0.0)) / (1.0 - gamma), float(torch.clamp(self.r_hi, min=0.0)) / (1.0 - gamma)

    # Inputs are clamped to +-10 normalised units: a state the policy pushes far outside the data
    # then cannot make the networks extrapolate without bound (the gradient there is zero).
    def norm_s(self, s):
        return ((s - self.s_mean) / self.s_std).clamp(-10.0, 10.0)

    def norm_c(self, c):
        return ((c - self.c_mean) / self.c_std).clamp(-10.0, 10.0)

    def _analytic(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """The analytically known part of the next state (zeros elsewhere), differentiable in s and a."""
        out = torch.zeros_like(s)
        ac = a.clamp(-self.layout.action_bound, self.layout.action_bound)
        if self.i_adst.numel():
            out = out.index_copy(1, self.i_adst, ac.index_select(1, self.i_asrc))
        if self.i_sdst.numel():
            out = out.index_copy(1, self.i_sdst, s.index_select(1, self.i_ssrc))
        if self.i_hold.numel():
            out = out.index_copy(1, self.i_hold, s.index_select(1, self.i_hold))
        return out

    def forward(self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor):
        """One step. Returns (next state, reward, termination logit)."""
        ns, nc = self.norm_s(s), self.norm_c(c)
        s2 = s + self.dyn(torch.cat([ns, a, nc], -1)) * self.d_std
        s2 = torch.where(self.analytic, self._analytic(s, a), s2)
        ns2 = self.norm_s(s2)
        r = self.rew(torch.cat([ns, a, ns2, nc], -1)).squeeze(-1) * self.r_std + self.r_mean
        r = torch.maximum(torch.minimum(r, self.r_hi), self.r_lo)   # never beyond what the simulator has paid
        logit = self.term(torch.cat([ns2, nc], -1)).squeeze(-1)
        return s2, r, logit

    def window_loss(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, term: torch.Tensor,
                    done: torch.Tensor, c: torch.Tensor):
        """SuperMimic Eq. 1 on k-step windows, rolled out autoregressively from the logged first state.

        s (B, k+1, S), a (B, k, A), r/term/done (B, k), c (B, C). Transition j counts while no episode
        ended before it; its next-state term also needs the episode not to end at j (the logged next
        state is then a reset state)."""
        B, k = a.shape[:2]
        alive = torch.ones(B, device=s.device)
        x = s[:, 0]
        ls = lr = lt = torch.zeros((), device=s.device)
        n_s = n_t = torch.zeros((), device=s.device)
        for j in range(k):
            x, rh, logit = self(x, a[:, j], c)
            m_s = alive * (1.0 - done[:, j])
            err = ((x - s[:, j + 1]) / self.s_std).square().mean(-1)
            ls = ls + (err * m_s).sum()
            lr = lr + ((((rh - r[:, j]) / self.r_std) ** 2) * alive).sum()
            lt = lt + (nn.functional.binary_cross_entropy_with_logits(logit, term[:, j], reduction="none") * alive).sum()
            n_s, n_t = n_s + m_s.sum(), n_t + alive.sum()
            alive = alive * (1.0 - done[:, j])
            # after an episode end the logged state is a reset: restart the rollout there (masked anyway)
            x = torch.where((alive > 0).unsqueeze(-1), x, s[:, j + 1])
        ls, lr, lt = ls / n_s.clamp(min=1), lr / n_t.clamp(min=1), lt / n_t.clamp(min=1)
        return ls + lr + lt, {"wm/state": ls.detach(), "wm/reward": lr.detach(), "wm/term": lt.detach()}
