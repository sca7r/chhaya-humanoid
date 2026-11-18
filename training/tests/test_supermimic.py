"""SuperMimic core (training/supermimic) on a small batched torch env: no simulator needed."""
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from supermimic import Layout, SuperMimic, WorldModel, tiny  # noqa: E402
from supermimic.replay import Replay  # noqa: E402


class VelocityToy:
    """A body that must match a commanded 2D velocity. State [vel 2, cmd 2, prev action 2] (no position:
    like Chhaya's state, every dimension stays bounded);
    sim context = actuator gain (randomised per episode, like servo strength). Terminates on a runaway
    speed (> 2 m/s), times out after 200 steps. Same interface as the Chhaya adapter."""

    def __init__(self, n=64, device="cpu"):
        self.n, self.device = n, torch.device(device)
        self.state_dim, self.action_dim, self.ctx_dim = 6, 2, 1
        self.layout = Layout(obs=[0, 1, 2, 3, 4, 5], obs_noise_std=[0.02, 0.02, 0, 0, 0, 0],
                             action_dst=[4, 5], action_src=[0, 1], hold=[2, 3], history=3)
        self.s = torch.zeros(n, 6); self.g = torch.ones(n, 1); self.t = torch.zeros(n)
        self.hist = torch.zeros(n, 3, 6)

    def _reset(self, m):
        k = int(m.sum())
        self.s[m] = 0.0
        self.s[m, 2:4] = torch.rand(k, 2) * 2 - 1
        self.g[m] = 0.7 + 0.6 * torch.rand(k, 1)
        self.t[m] = 0
        self.hist[m] = 0.0
        self._push(m)

    def _push(self, m):
        o = self.s.clone() + torch.randn(self.n, 6) * torch.tensor(self.layout.obs_noise_std)
        self.hist[m] = torch.cat([self.hist[m, 1:], o[m].unsqueeze(1)], dim=1)

    def reset(self):
        self._reset(torch.ones(self.n, dtype=torch.bool))

    def state(self):
        return self.s.clone()

    def context(self):
        return self.g.clone()

    def policy_obs(self):
        return self.hist.reshape(self.n, -1).clone()

    def step(self, a):
        a = a.clamp(-1, 1)
        v = 0.8 * self.s[:, 0:2] + 0.4 * self.g * a
        r = torch.exp(-(v - self.s[:, 2:4]).square().sum(-1) / 0.1) - 0.05 * (a - self.s[:, 4:6]).square().sum(-1)
        self.s[:, 0:2] = v
        self.s[:, 4:6] = a
        self.t += 1
        term = self.s[:, 0:2].abs().max(-1).values > 2.0
        trunc = (self.t >= 200) & ~term
        self._push(torch.ones(self.n, dtype=torch.bool))
        done = term | trunc
        if done.any():
            self._reset(done)
        return r, term, trunc


def _eval(actor, env, steps=150):
    env.reset()
    tot = 0.0
    with torch.no_grad():
        for _ in range(steps):
            r, _, _ = env.step(actor(env.policy_obs()))
            tot += float(r.mean())
    return tot / steps


def test_analytic_slots_are_exact_and_differentiable():
    env = VelocityToy(4)
    wm = WorldModel(6, 2, 1, env.layout, (16,), (8,))
    s = torch.randn(4, 6)
    a = torch.randn(4, 2, requires_grad=True)
    s2, r, logit = wm(s, a, torch.ones(4, 1))
    assert torch.equal(s2[:, 4:6], a.detach().clamp(-1, 1))     # previous-action slots = clamp(action)
    assert torch.equal(s2[:, 2:4], s[:, 2:4])                    # command held
    s2[:, 4:6].sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


def test_replay_masks_episode_boundaries():
    rp = Replay(1, 50, 3, 1, 1, "cpu")
    for t in range(20):
        done = torch.tensor([t == 9])
        rp.add(torch.full((1, 3), float(t)), torch.zeros(1, 1), torch.zeros(1), done, done, torch.zeros(1, 1))
    # history for the state stored at step 11 (second step of episode 2): frames 9 and earlier are zeros
    torch.manual_seed(0)
    rp._pick = lambda B, lo, hi: (torch.tensor([11]), torch.tensor([0]))
    s, c, hist = rp.seeds(1, 4, torch.tensor([0, 1, 2]), torch.zeros(3))
    assert float(s[0, 0]) == 11.0
    assert hist[0, :, 0].tolist() == [0.0, 0.0, 10.0, 11.0]
    # a window across the boundary: the world-model loss masks everything after the episode end
    rp._pick = lambda B, lo, hi: (torch.tensor([8]), torch.tensor([0]))
    s, a, r, term, done, c = rp.windows(1, 4)
    assert s[0, :, 0].tolist() == [8.0, 9.0, 10.0, 11.0, 12.0] and done[0].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_policy_learns_through_the_world_model():
    torch.manual_seed(0)
    env, ev = VelocityToy(64), VelocityToy(64)
    sm = SuperMimic(env, tiny())
    before = _eval(sm.actor, ev)
    logs = {}
    sm.train(iterations=400, on_log=lambda it, l: logs.update(l))
    after = _eval(sm.actor, ev)
    assert logs["wm/state"] < 0.05, logs                        # the world model predicts the toy
    # mean reward per step, 1.0 = perfect tracking. Measured 2026-11-18 (seeds 0-3): 0.07-0.09 before,
    # 0.976-0.978 after; the exact dynamics in place of the learned model give 0.95 in 200 iterations.
    assert after > 0.85 and after > before + 0.6, (before, after, logs)
