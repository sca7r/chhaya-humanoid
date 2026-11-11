"""Get-up / fall-recovery reward kernels (envs/isaaclab/getup_env.py). Pure torch: no Isaac, unit-tested on the CPU
(tests/test_getup_rewards.py). Every term is >= 0; the sign lives in the weight (rewards.REWARD_WEIGHT_SIGN).

Quantities (all ground truth, never the noisy observation):
    up      = -g_z, the base's uprightness from the projected gravity: +1 upright, 0 on its side / back / front,
              -1 upside down. (The walking `upright` term only reads g_x, g_y and scores upside down as upright.)
    h_hat   = clamp((h - h_floor) / (h_stand - h_floor), 0, 1): base height above the floor, 0 lying, 1 standing at
              the home pose (h_stand = the spawn height, h_floor = the fall bank's median lying height).
    standing = h_hat > stand_height_frac AND up > cos(stand_tilt) AND both feet loaded. The success / hand-off test.

Staged like HoST (Huang et al., RSS 2025): righting (getup_upright) -> rising (getup_height, gated by up) -> on the
feet (getup_feet_support) -> holding the walking policy's home pose still (getup_stand, getup_still). The stand bonus
is the largest term, so the fastest safe get-up wins: every second standing earlier earns it for longer.

Posture once up (2026-11-02). Run 2026-11-01_18-59-35 stood wide, low and on its sole edges, then sank into a crouch;
the stand terms now use the SOFT stand_score (no cliff at the height threshold). Pose and sole flatness are QUALITY
FACTORS in [0, 1] that scale non-negative stand bonuses (getup_stand, getup_feet_flat), never penalties. So for any
pose held still, rising never lowers the reward (no valley by construction; tests/test_getup_rewards.py checks it).
Penalties on the pose (a posture-gated sum (q - home)^2 and the walking foot_flat_l1, run 2026-11-02_11-45-46) put a
valley across the rise: the joints are necessarily far from home while climbing (|q - home|^2 ~ 2 rad^2 at h_hat
0.65, scored on run 1's real get-ups in MuJoCo), so a squat at h_hat ~0.55 scored 2.4/step, the climb 0.7 and
standing 1.3. It squatted there for 233M steps.
"""
from __future__ import annotations

import math

import torch

from .rewards import foot_flat_roll_l1


def uprightness(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """(N,) up = -g_z in [-1, 1]."""
    return -projected_gravity_b[:, 2]


def height_progress(base_height: torch.Tensor, h_floor: float, h_stand: float) -> torch.Tensor:
    """(N,) h_hat in [0, 1]."""
    return torch.clamp((base_height - h_floor) / max(h_stand - h_floor, 1e-6), 0.0, 1.0)


def standing_mask(h_hat: torch.Tensor, up: torch.Tensor, feet_loaded: torch.Tensor,
                  height_frac: float, tilt_deg: float) -> torch.Tensor:
    """(N,) bool: high enough, upright enough and on BOTH feet. feet_loaded (N, 2) bool."""
    return (h_hat > height_frac) & (up > math.cos(math.radians(tilt_deg))) & feet_loaded.all(dim=-1)


def soft_ramp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """clamp((x - lo) / (hi - lo), 0, 1): 0 below lo, 1 above hi, linear between."""
    return torch.clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def stand_score(h_hat: torch.Tensor, up: torch.Tensor, feet_loaded: torch.Tensor,
                h_lo: float, h_hi: float, up_lo: float, up_hi: float) -> torch.Tensor:
    """(N,) soft 'standing' in [0, 1]: ramps with height (h_lo -> h_hi) and uprightness (up_lo -> up_hi), times the
    fraction of loaded feet. It is exactly 1 wherever standing_mask holds (h_hi / up_hi = its thresholds), and unlike
    the mask it has a gradient below them: run 2026-11-01_18-59-35 sank under the hard 0.9 height gate at ~215M and
    nothing paid it to come back up (getup_stand went to exactly 0)."""
    return soft_ramp(h_hat, h_lo, h_hi) * soft_ramp(up, up_lo, up_hi) * feet_loaded.float().mean(dim=-1)


def pose_quality(joint_pos: torch.Tensor, home: torch.Tensor, sharp_rad2: float, broad_rad2: float) -> torch.Tensor:
    """(N,) in (0, 1]: 0.5 exp(-E / sharp) + 0.5 exp(-E / broad), E = sum_j (q_j - home_j)^2 [rad^2]. The sharp kernel
    pulls into the hand-off box (every joint within 0.2 rad); the broad one keeps a gradient from far stances (run 1
    stood at E ~1.1 and crouched at E ~4.4, where the old single exp(-E / 0.5) was flat: 0.11 and 1e-4)."""
    err2 = torch.sum(torch.square(joint_pos - home), dim=-1)
    return 0.5 * torch.exp(-err2 / sharp_rad2) + 0.5 * torch.exp(-err2 / broad_rad2)


def foot_flat_quality(foot_quat_w: torch.Tensor, root_quat_w: torch.Tensor, in_contact: torch.Tensor,
                      sigma: float) -> torch.Tensor:
    """(N,) in (0, 1]: exp(-sum |sin(sole roll)| over loaded feet / sigma), the walking foot_flat_l1 geometry. Its slope
    at flat is 1 / sigma (no dead zone near flat, the L1 lesson of 2026-10-25). Run 1 stood on 21-27 deg rolled soles."""
    return torch.exp(-foot_flat_roll_l1(foot_quat_w, root_quat_w, in_contact) / sigma)


def getup_upright(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """((up + 1) / 2)^2: 1 upright, 0.25 lying on any side, 0 upside down. Smooth everywhere, so the righting phase
    (rolling off the back) has a gradient from every pose."""
    return torch.square(0.5 * (uprightness(projected_gravity_b) + 1.0))


def getup_height(h_hat: torch.Tensor, projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """h_hat * (up + 1) / 2: height only pays while the base is upright, so lifting the torso sideways is not a
    shortcut."""
    return h_hat * 0.5 * (uprightness(projected_gravity_b) + 1.0)


def getup_feet_support(projected_gravity_b: torch.Tensor, feet_loaded: torch.Tensor, min_up: float) -> torch.Tensor:
    """[up > min_up] * mean(feet loaded): once roughly upright, stand on the feet (not the knees or the base)."""
    gate = (uprightness(projected_gravity_b) > min_up).float()
    return gate * feet_loaded.float().mean(dim=-1)


def getup_stand(standing: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """standing * quality >= 0: standing in the walking policy's home pose (the hand-off). standing is a bool mask or
    the soft stand_score in [0, 1] (the env passes the soft score), quality = pose_quality."""
    return standing.float() * quality


def getup_feet_flat(standing: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """standing * quality >= 0, quality = foot_flat_quality: standing on flat soles, not their edges. A bonus, not the
    walking penalty, so a badly placed foot only costs part of the stand bonus and never makes rising a loss."""
    return standing.float() * quality


def getup_still(standing: torch.Tensor, lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor,
                joint_vel: torch.Tensor) -> torch.Tensor:
    """standing * (|v|^2 + 0.1 |w|^2 + 0.01 |qd|^2): once up, hold still (penalty). The env passes stand_score x
    pose_quality as `standing`: the climb itself is fast (0.3-0.5/step of this through h_hat 0.8-0.9 on run 1's
    get-ups when gated by the stand score alone), so only holding the home pose is asked to be still."""
    motion = (torch.sum(torch.square(lin_vel_b), dim=-1) + 0.1 * torch.sum(torch.square(ang_vel_b), dim=-1)
              + 0.01 * torch.sum(torch.square(joint_vel), dim=-1))
    return standing.float() * motion


def overload_step(state: torch.Tensor, torque: torch.Tensor, stall_nm: float, threshold_frac: float,
                  leak_s: float, dt: float) -> torch.Tensor:
    """Leaky integrator per servo, mimicking the STS3215 overload protection (it cuts the output after sustained
    high current): z <- z * exp(-dt / leak) + dt * (|tau| / stall - threshold)_+. Returns the new state (N, J)."""
    excess = torch.clamp(torque.abs() / stall_nm - threshold_frac, min=0.0)
    return state * math.exp(-dt / max(leak_s, 1e-6)) + dt * excess


def servo_overload(state: torch.Tensor) -> torch.Tensor:
    """sum_j z_j of overload_step's state (penalty)."""
    return torch.sum(state, dim=-1)


def body_impact(base_force_n: torch.Tensor, weight_n: float, mult: float, clip: float = 25.0) -> torch.Tensor:
    """((F_base - mult * m g)_+ / m g)^2, clipped: hard landings on the base (the Jetson and the IMU live there).
    Lying on the base loads it with < 1 m g, which is free."""
    x = torch.clamp(base_force_n - mult * weight_n, min=0.0) / weight_n
    return torch.clamp(torch.square(x), max=clip)


def ang_vel_limit(ang_vel_b: torch.Tensor, limit: float, clip: float = 25.0) -> torch.Tensor:
    """(|w| - limit)_+^2, clipped: no violent flips (gyro, IMU fusion and the servos' horns all prefer that)."""
    return torch.clamp(torch.square(torch.clamp(torch.norm(ang_vel_b, dim=-1) - limit, min=0.0)), max=clip)


def self_contact_l2(forces_n: torch.Tensor, thresh_n: float = 0.5, sat_n: float = 5.0) -> torch.Tensor:
    """Leg-on-leg pressing (floor part removed), saturating: same kernel as rewards.leg_contact_l2."""
    x = torch.clamp((forces_n - thresh_n) / max(sat_n - thresh_n, 1e-6), min=0.0, max=1.0)
    return torch.sum(torch.square(x), dim=-1)
