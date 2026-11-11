"""
Arm targets for Chhaya, while a LEGS-ONLY policy walks.

The walking policy neither sees nor commands the 8 arm joints. They are servo-driven (same BAM
model as the legs) toward targets produced here, so that during training the policy meets every
kind of arm motion the real robot will make (scripted gestures, arm swing, holding a pose) and
learns to treat the arms as a moving payload.

Per env, one MODE is drawn at every reset:
  rest    the arm rest pose + a small constant per-episode offset (what the robot does most)
  swing   counter-swing to the legs, synced to the gait clock, random amplitude
  random  smooth band-limited random motion inside safe ranges
  hold    random poses inside safe ranges, re-drawn every few seconds, smoothstep-blended
All targets are rate-limited to what the STS3215 can follow and clamped to the joint limits.

Pure torch: no Isaac Lab import, so it is unit-tested on the CPU (tests/test_arm_motion.py).
Joint order is the order of `joint_names` given to the constructor (L,R interleaved per type).
"""
from __future__ import annotations

import math

import torch

MODES = ("rest", "swing", "random", "hold")

# Both arms share sign conventions: +roll = arm out, -pitch = forward/up, -elbow = flex, +yaw = outward.
REST = {"shoulder_pitch": 0.0, "shoulder_roll": 0.12, "arm_yaw": 0.0, "elbow": -0.35}
SAFE = {"shoulder_pitch": (-2.0, 0.6), "shoulder_roll": (0.08, 1.2), "arm_yaw": (-0.8, 0.8), "elbow": (-1.8, 0.0)}

DEFAULTS = {
    "mode_probs": {"rest": 0.3, "swing": 0.3, "random": 0.25, "hold": 0.15},
    "rest_offset_rad": 0.12,          # +- per joint, constant per episode
    "swing_amp_range": (0.1, 0.6),    # shoulder pitch amplitude [rad]
    "random_bandwidth_hz": (0.3, 1.5),
    "random_amplitude": 0.7,          # fraction of each joint's safe half-range
    "hold_period_s": (1.0, 3.0),
    "max_speed_rad_s": 4.0,           # STS3215 no-load speed is ~4.9 rad/s
}


def _joint_type(name: str) -> str:
    for t in REST:
        if name[2:].startswith(t):
            return t
    raise ValueError(f"arm joint {name!r}: expected l_/r_ + one of {list(REST)}")


class ArmMotion:
    def __init__(self, num_envs: int, joint_names: list[str], lower: torch.Tensor, upper: torch.Tensor,
                 cfg: dict | None, device, generator: torch.Generator | None = None):
        self.n, self.device = num_envs, device
        self.names = list(joint_names)
        self.cfg = {**DEFAULTS, **(cfg or {})}
        probs = {**DEFAULTS["mode_probs"], **(self.cfg.get("mode_probs") or {})}
        unknown = set(probs) - set(MODES)
        if unknown:
            raise ValueError(f"arm_motion.mode_probs: unknown mode(s) {sorted(unknown)}; expected {MODES}")
        p = torch.tensor([float(probs.get(m, 0.0)) for m in MODES])
        if p.sum() <= 0:
            raise ValueError("arm_motion.mode_probs must have a positive sum")
        self.mode_p = (p / p.sum()).to(device)
        types = [_joint_type(n) for n in self.names]
        self.side = torch.tensor([1.0 if n.startswith("l_") else -1.0 for n in self.names], device=device)
        self.rest = torch.tensor([REST[t] for t in types], device=device)
        safe = torch.tensor([SAFE[t] for t in types], device=device)
        lo = torch.maximum(safe[:, 0], lower.to(device))
        hi = torch.minimum(safe[:, 1], upper.to(device))
        self.safe_lo, self.safe_hi = lo, hi
        self.lower, self.upper = lower.to(device), upper.to(device)
        self.is_pitch = torch.tensor([t == "shoulder_pitch" for t in types], device=device)
        self.is_elbow = torch.tensor([t == "elbow" for t in types], device=device)
        self.g = generator
        J = len(self.names)
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.offset = torch.zeros(num_envs, J, device=device)
        self.amp = torch.zeros(num_envs, device=device)
        self.bw = torch.ones(num_envs, device=device)
        self.x = torch.zeros(num_envs, J, device=device)      # random-mode filter state
        self.v = torch.zeros(num_envs, J, device=device)
        self.hold_from = self.rest.repeat(num_envs, 1)
        self.hold_to = self.rest.repeat(num_envs, 1)
        self.hold_t = torch.zeros(num_envs, device=device)
        self.hold_T = torch.ones(num_envs, device=device)
        self.targets = self.rest.repeat(num_envs, 1)

    # ------------------------------------------------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.g)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self.g)

    def _uniform(self, n, rng):
        lo, hi = float(rng[0]), float(rng[1])
        return lo + (hi - lo) * self._rand(n)

    def _random_pose(self, n):
        return self.safe_lo + (self.safe_hi - self.safe_lo) * self._rand(n, len(self.names))

    def reset(self, env_ids: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = ids.numel()
        if n == 0:
            return
        self.mode[ids] = torch.multinomial(self.mode_p, n, replacement=True, generator=self.g)
        r = float(self.cfg["rest_offset_rad"])
        self.offset[ids] = (2 * self._rand(n, len(self.names)) - 1) * r
        self.amp[ids] = self._uniform(n, self.cfg["swing_amp_range"])
        self.bw[ids] = self._uniform(n, self.cfg["random_bandwidth_hz"])
        self.x[ids] = 0.0
        self.v[ids] = 0.0
        start = torch.clamp(self.rest + self.offset[ids], self.safe_lo, self.safe_hi)
        self.hold_from[ids] = start
        self.hold_to[ids] = self._random_pose(n)
        self.hold_t[ids] = 0.0
        self.hold_T[ids] = self._uniform(n, self.cfg["hold_period_s"])
        self.targets[ids] = start

    def step(self, dt: float, gait_phase: torch.Tensor) -> torch.Tensor:
        """Advance one control step; gait_phase in radians [0, 2*pi) (BipedEnv.gait_phase)."""
        rest = torch.clamp(self.rest + self.offset, self.safe_lo, self.safe_hi)
        # swing: left arm forward (-pitch) while the right leg swings forward, elbows flex on the forward swing
        s = torch.sin(gait_phase).unsqueeze(1)
        a = self.amp.unsqueeze(1)
        swing = rest + self.is_pitch * (self.side * a * s) \
            - self.is_elbow * (0.25 * a * torch.clamp(-self.side * s, min=0.0))
        # random: critically damped 2nd-order filter on white noise (smooth position and velocity)
        w = (2 * math.pi * self.bw).unsqueeze(1)
        self.v += dt * (w * w * (1.8 * self._randn(*self.x.shape) - self.x) - 2 * w * self.v)
        self.x += dt * self.v
        mid, half = (self.safe_lo + self.safe_hi) / 2, (self.safe_hi - self.safe_lo) / 2
        rnd = mid + float(self.cfg["random_amplitude"]) * half * torch.tanh(self.x)
        # hold: blend between random poses
        self.hold_t += dt
        done = self.hold_t >= self.hold_T
        if bool(done.any()):
            k = int(done.sum())
            self.hold_from[done] = self.hold_to[done]
            self.hold_to[done] = self._random_pose(k)
            self.hold_t[done] = 0.0
            self.hold_T[done] = self._uniform(k, self.cfg["hold_period_s"])
        u = torch.clamp(self.hold_t / self.hold_T, 0.0, 1.0).unsqueeze(1)
        u = u * u * (3 - 2 * u)
        hold = self.hold_from + (self.hold_to - self.hold_from) * u

        m = self.mode.unsqueeze(1)
        want = torch.where(m == 0, rest, torch.where(m == 1, swing, torch.where(m == 2, rnd, hold)))
        dmax = float(self.cfg["max_speed_rad_s"]) * dt
        self.targets = torch.clamp(self.targets + torch.clamp(want - self.targets, -dmax, dmax),
                                   self.lower, self.upper)
        return self.targets
