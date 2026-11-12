"""Arm motions: gait-synchronised arm swing, gestures and smooth random motion.

Pure numpy and simulator-agnostic: every function returns a dict {joint_name: angle_rad} for
the 8 arm joints (chhaya.model.ARM_JOINTS). The same code drives
  * the simulation (sim backend, training randomisation),
  * the real robot later (STS3215 bus, or scripted cheap servos).

Sign conventions (identical for both arms): -pitch = arm forward/up, +roll = arm out,
-elbow = flex, +yaw = upper arm rotates outward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from chhaya.model import ARM_JOINTS, arm_limits

REST = {"l_shoulder_pitch": 0.0, "r_shoulder_pitch": 0.0,
        "l_shoulder_roll": 0.20, "r_shoulder_roll": 0.20,
        "l_arm_yaw": 0.0, "r_arm_yaw": 0.0,
        "l_elbow": -0.35, "r_elbow": -0.35}

POSES = {
    "rest": REST,
    "zero": {j: 0.0 for j in ARM_JOINTS},
    "t_pose": {**REST, "l_shoulder_roll": 1.57, "r_shoulder_roll": 1.57, "l_elbow": 0.0, "r_elbow": 0.0},
    "hands_forward": {**REST, "l_shoulder_pitch": -1.4, "r_shoulder_pitch": -1.4, "l_elbow": -0.3, "r_elbow": -0.3},
    "hands_up": {**REST, "l_shoulder_pitch": -2.9, "r_shoulder_pitch": -2.9, "l_elbow": -0.2, "r_elbow": -0.2},
    "balance": {**REST, "l_shoulder_roll": 0.6, "r_shoulder_roll": 0.6, "l_elbow": -0.5, "r_elbow": -0.5},
}


def clip(pose: dict) -> dict:
    return {j: float(np.clip(v, *arm_limits(j))) for j, v in pose.items()}


def blend(a: dict, b: dict, s: float) -> dict:
    s = float(np.clip(s, 0.0, 1.0))
    s = s * s * (3 - 2 * s)                     # smoothstep: zero velocity at both ends
    return {j: (1 - s) * a[j] + s * b[j] for j in a}


def arm_swing(phase: float, amplitude: float = 0.35, base: dict | None = None) -> dict:
    """Counter-swing to the legs. phase in [0, 1): 0 = left foot touchdown (same convention as
    the training repo's gait clock). Left arm swings forward while the right leg swings forward."""
    base = base or REST
    s = math.sin(2 * math.pi * phase)
    p = dict(base)
    p["l_shoulder_pitch"] = base["l_shoulder_pitch"] + amplitude * s
    p["r_shoulder_pitch"] = base["r_shoulder_pitch"] - amplitude * s
    # elbows flex a little more on the forward swing, like people do
    p["l_elbow"] = base["l_elbow"] - 0.25 * amplitude * max(0.0, -s)
    p["r_elbow"] = base["r_elbow"] - 0.25 * amplitude * max(0.0, s)
    return clip(p)


def wave(t: float, side: str = "r", freq: float = 1.6) -> dict:
    p = dict(REST)
    p[f"{side}_shoulder_pitch"] = -2.6
    p[f"{side}_shoulder_roll"] = 0.35
    p[f"{side}_arm_yaw"] = 0.3
    p[f"{side}_elbow"] = -1.0 + 0.45 * math.sin(2 * math.pi * freq * t)
    return clip(p)


@dataclass
class Gesture:
    """Keyframe sequence [(time_s, pose_dict)], smoothstep-blended, optional loop."""
    keys: list
    loop: bool = False

    def __call__(self, t: float) -> dict:
        T = self.keys[-1][0]
        if self.loop and T > 0:
            t = t % T
        if t <= self.keys[0][0]:
            return dict(self.keys[0][1])
        for (t0, a), (t1, b) in zip(self.keys, self.keys[1:]):
            if t <= t1:
                return blend(a, b, (t - t0) / max(t1 - t0, 1e-6))
        return dict(self.keys[-1][1])


GESTURES = {
    "wave": lambda t: wave(t),
    "raise_and_lower": Gesture([(0, REST), (1.2, POSES["hands_up"]), (2.0, POSES["hands_up"]), (3.2, REST)], loop=True),
    "reach": Gesture([(0, REST), (1.0, POSES["hands_forward"]), (1.6, POSES["hands_forward"]), (2.6, REST)], loop=True),
    "t_pose": Gesture([(0, REST), (1.0, POSES["t_pose"]), (2.0, POSES["t_pose"]), (3.0, REST)], loop=True),
}


@dataclass
class RandomArms:
    """Smooth random arm targets for training robustness: each joint follows a low-passed random
    walk inside a safe sub-range of its limits. Use it to train the leg policy with arms that
    move like the scripted ones will (or worse), so the walking policy treats them as a payload."""
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    bandwidth_hz: float = 0.8
    amplitude: float = 0.6          # fraction of each joint's safe half-range
    _x: np.ndarray | None = None
    _v: np.ndarray | None = None

    SAFE = {"shoulder_pitch": (-2.0, 0.6), "shoulder_roll": (0.0, 1.2), "arm_yaw": (-0.8, 0.8), "elbow": (-1.8, 0.0)}

    def reset(self):
        self._x = np.zeros(len(ARM_JOINTS))
        self._v = np.zeros(len(ARM_JOINTS))

    def step(self, dt: float) -> dict:
        if self._x is None:
            self.reset()
        w = 2 * math.pi * self.bandwidth_hz
        # critically damped second-order filter on white noise: smooth position AND velocity
        self._v += dt * (w * w * (self.rng.normal(size=self._x.shape) * 1.8 - self._x) - 2 * w * self._v)
        self._x += dt * self._v
        out = {}
        for i, j in enumerate(ARM_JOINTS):
            lo, hi = self.SAFE[j[2:]]
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            out[j] = float(np.clip(mid + self.amplitude * half * np.tanh(self._x[i]), lo, hi))
        return out
