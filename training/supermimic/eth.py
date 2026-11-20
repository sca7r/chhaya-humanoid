"""SuperMimic the way the official code and ETH's terrain-grounded imagination do it: the world model
learns only the robot's dynamics; reward and termination are computed analytically from the
predicted state, so the policy gradient goes through exact formulas instead of a learned reward head.

  * Dynamics: the MLP predicts the next velocities (body-frame linear and angular, joint), as a
    normalised change; joint angles, base height and the gravity direction are then integrated from
    them (semi-implicit, ``integrate``), so every prediction is kinematically consistent. The noisy,
    delayed sensor copies in the observation block are still predicted directly (as in v1).
  * Reward: a velocity-tracking reward in the style of the paper's Table A.6 and ETH's RewardCfg,
    every term times the control period dt. Feet come from forward kinematics of the URDF (``LegFK``).
  * Termination: the env's tilt rule (45 deg), as a sigmoid so the survival weight stays differentiable.
  * Loss: each physical quantity is scaled by a nominal error (Bryson's rule), the rest by its std.

Refs: arXiv:2610.28480 (Eq. 1-3, Table A.6); github.com/HybridRobotics/supermimic (world_model_env.py);
github.com/leggedrobotics/terrain-grounded-imagination (tgi/mdp/mdp.py).
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch
from torch import nn

from .world_model import Layout, WorldModel


def mlp_ln(inp: int, hidden, out: int) -> nn.Sequential:
    """Linear -> LayerNorm -> GELU blocks, as in the official models.build_mlp."""
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def _rpy(r: float, p: float, y: float) -> torch.Tensor:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    """(B,) angles about a fixed unit axis -> (B, 3, 3) (Rodrigues)."""
    x, y, z = axis.tolist()
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], device=th.device)
    s, c = th.sin()[:, None, None], th.cos()[:, None, None]
    return torch.eye(3, device=th.device) + s * K + (1 - c) * (K @ K)


class LegFK:
    """Positions of the foot-link origins in the base frame, from the URDF, differentiable in q."""

    def __init__(self, urdf: str, joint_names, feet=("l_foot_link_1", "r_foot_link_1"), base="base_link"):
        joints = {j.find("child").get("link"): j for j in ET.parse(urdf).getroot().findall("joint")}
        self.chains = []
        for foot in feet:
            chain, link = [], foot
            while link != base:
                j = joints[link]
                o = j.find("origin")
                xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
                rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
                ax = j.find("axis")
                axis = torch.tensor([float(v) for v in ax.get("xyz").split()]) if ax is not None else torch.tensor([1.0, 0, 0])
                idx = list(joint_names).index(j.get("name")) if j.get("type") != "fixed" else -1
                chain.append((torch.tensor(xyz), _rpy(*rpy), axis / axis.norm(), idx))
                link = j.find("parent").get("link")
            self.chains.append(chain[::-1])

    def __call__(self, q: torch.Tensor, normals: bool = False):
        """q (B, J) absolute joint angles in ``joint_names`` order -> (B, n_feet, 3); with ``normals`` also the
        foot-link z axes (B, n_feet, 3) = sole normals (the URDF's foot frames are level at q = 0)."""
        B, out, nz = q.shape[0], [], []
        for chain in self.chains:
            R = torch.eye(3, device=q.device).expand(B, 3, 3)
            p = torch.zeros(B, 3, device=q.device)
            for xyz, Ro, axis, idx in chain:
                p = p + R @ xyz.to(q.device)
                R = R @ Ro.to(q.device)
                if idx >= 0:
                    R = R @ _axis_angle(axis, q[:, idx])
            out.append(p)
            nz.append(R[:, :, 2])
        return (torch.stack(out, 1), torch.stack(nz, 1)) if normals else torch.stack(out, 1)


@dataclass
class RewardCfg:
    """Weights for Chhaya (0.3 m tall, commands up to 0.2 m/s); terms as in ETH's RewardCfg / Table A.6."""
    w_lin: float = 2.0
    std_lin: float = 0.2            # the env's track_lin_vel_xy_exp std
    w_ang: float = 1.5
    std_ang: float = 0.3
    w_upright: float = 0.5
    std_upright: float = math.sqrt(0.05)
    w_pose: float = 2.0             # standing commands only
    std_pose: float = 0.2
    w_limits: float = -1.0
    w_action_rate: float = -0.1
    w_lin_z: float = -1.0
    w_ang_xy: float = -0.05
    w_clearance: float = -2.0       # swing feet below the target, times their horizontal speed
    clearance_target: float = 0.025  # the env's clock_swing_height_m
    w_slip: float = -1.0
    w_gait: float = 0.5             # the env's gait_phase_contact: right foot down when sin(phase) > 0, left when < 0
    w_still: float = -0.5           # standing commands: per foot off the ground
    w_dev_hip: float = -0.5         # hip roll, hip yaw and ankle roll, L1
    w_dev_arms: float = -0.1
    # arm joint speed^2: the arms twisted at ~1.6 rad/s RMS for nothing, ~1.8 N m RMS through the servos' reflected
    # rotor inertia (above the STS3215's continuous rating). -0.02 makes that waggle cost ~0.2 per step.
    w_arm_vel: float = -0.02
    # Human arm swing (with a gait table): each shoulder pitch follows the OPPOSITE hip's table pitch, scaled
    # (left arm forward with the right leg). Cancels the legs' yaw twist. Hip and shoulder pitch share the sign
    # convention (negative = forward, checked by forward kinematics). Shoulder pitch is left out of the arm
    # speed / deviation penalties, which keep arm yaw, roll and elbow quiet.
    w_arm_swing: float = 0.5
    arm_swing_gain: float = 0.6
    std_arm_swing: float = 0.15
    w_imit_q: float = 1.0           # with a gait table (env `reference:`): leg joints, sole heights and contacts
    w_imit_h: float = 1.0           # of the table at (command, phase); the contact term replaces w_gait
    w_imit_c: float = 0.5
    w_flat_l2: float = -5.0         # the env's foot_flat_l2 / foot_flat_l1: sole roll of feet on the ground
    w_flat_l1: float = -3.0
    cmd_threshold: float = 0.02
    walking_threshold: float = 0.05
    contact_band: float = 0.004     # sigmoid width [m] of the soft foot contact
    tilt_limit: float = 0.784       # the env's _tilt_exceeded
    tilt_softness: float = 0.03


class ChhayaMDP:
    """Reward, soft termination and integration on the adapter's world-model state (``Layout.phys``)."""

    def __init__(self, ph: dict, device, cfg: RewardCfg = RewardCfg()):
        self.cfg, self.dt, dev = cfg, ph["dt"], device
        self.sub = (ph["decimation"] + 1) / (2.0 * ph["decimation"])     # mean weight of the new velocity over the substeps
        self.iv, self.ih, self.iq, self.iqd = ph["v_b"], ph["h"], ph["q"], ph["qd"]
        self.iw, self.ig, self.icmd, self.iph = ph["w_b"], ph["g_b"], ph["cmd"], ph["phase"]
        self.adst, self.asrc = ph["action_dst"], ph["action_src"]
        self.q0 = torch.tensor(ph["q0"], device=dev)
        self.lo, self.hi = torch.tensor(ph["soft_lo"], device=dev), torch.tensor(ph["soft_hi"], device=dev)
        names = ph["joint_names"]
        # hip roll/yaw and ankle roll: the standing A-frame (hips rolled in, ankles out, soles on their edges; the MuJoCo
        # gate's edge-stance check) is a deviation of exactly these
        self.m_hip = torch.tensor([1.0 if any(k in n for k in ("hip_roll", "hip_yaw", "ankle_roll")) else 0.0 for n in names], device=dev)
        self.m_arm = torch.tensor([1.0 if any(k in n for k in ("shoulder_roll", "arm_yaw", "elbow")) else 0.0 for n in names], device=dev)
        self.names = list(names)
        self.fk = LegFK(ph["urdf"], names)
        self.sole = ph.get("sole_height", 0.0)
        self.ref = None                                          # gait table, set by the adapter (attach_reference)
        self.max_step_reward = self.dt * (cfg.w_lin + cfg.w_ang + cfg.w_upright + cfg.w_pose + cfg.w_gait
                                          + cfg.w_imit_q + cfg.w_imit_h + cfg.w_arm_swing)

    def _sl(self, s, idx):
        return s[:, idx[0]:idx[1]]

    def integrate(self, s: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Overwrite q, h and g in the predicted state from the predicted velocities."""
        dt, c = self.dt, self.sub
        qd, qd2 = self._sl(s, self.iqd), self._sl(s2, self.iqd)
        v, v2 = self._sl(s, self.iv), self._sl(s2, self.iv)
        w, w2 = self._sl(s, self.iw), self._sl(s2, self.iw)
        g = self._sl(s, self.ig)
        q2 = self._sl(s, self.iq) + dt * (qd + c * (qd2 - qd))
        wm, vm = w + c * (w2 - w), v + c * (v2 - v)
        g2 = g - dt * torch.cross(wm, g, dim=-1)               # a world-fixed vector seen from the turning body
        g2 = g2 / g2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        h2 = self._sl(s, self.ih) - dt * (g * vm).sum(-1, keepdim=True)   # world z axis in the body is -g
        out = s2.clone()
        out[:, self.iq[0]:self.iq[1]], out[:, self.ig[0]:self.ig[1]], out[:, self.ih[0]:self.ih[1]] = q2, g2, h2
        return out

    def feet(self, s: torch.Tensor):
        """Foot positions in the base frame (B, 2, 3) and their heights above the sole's rest height (B, 2)."""
        f, n = self.fk(self._sl(s, self.iq) + self.q0, normals=True)
        g = self._sl(s, self.ig)
        z = self._sl(s, self.ih) - (f * g.unsqueeze(1)).sum(-1)
        # sole roll as the env measures it: normal along the robot's horizontal lateral direction
        lat = torch.zeros_like(g); lat[:, 1] = 1.0
        lat = lat - (lat * g).sum(-1, keepdim=True) * g
        lat = lat / lat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._roll = (n * lat.unsqueeze(1)).sum(-1)              # (B, feet) sin(roll), read by reward()
        return f, z - self.sole

    def reward(self, s: torch.Tensor, a: torch.Tensor, s2: torch.Tensor, terms: bool = False):
        cf = self.cfg
        v, w, g = self._sl(s2, self.iv), self._sl(s2, self.iw), self._sl(s2, self.ig)
        q, cmd = self._sl(s2, self.iq) + self.q0, self._sl(s2, self.icmd)
        a_prev = torch.zeros_like(a)
        a_prev[:, self.asrc] = s[:, self.adst]
        speed = cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()
        active = (speed > cf.cmd_threshold).float()
        stand = 1.0 - active
        walking = (speed > cf.walking_threshold).float()

        r_lin = torch.exp(-((cmd[:, :2] - v[:, :2]) ** 2).sum(-1) / cf.std_lin ** 2)
        r_ang = torch.exp(-(cmd[:, 2] - w[:, 2]) ** 2 / cf.std_ang ** 2)
        r_up = torch.exp(-(g[:, :2] ** 2).sum(-1) / cf.std_upright ** 2)
        r_pose = torch.exp(-(((q - self.q0) / cf.std_pose) ** 2).mean(-1)) * stand
        r_lim = (torch.relu(q - self.hi) + torch.relu(self.lo - q)).sum(-1)
        r_rate = ((a.clamp(-1, 1) - a_prev) ** 2).sum(-1)
        r_lin_z = (g * v).sum(-1) ** 2
        r_ang_xy = (w ** 2).sum(-1) - (g * w).sum(-1) ** 2      # angular speed about horizontal axes
        dev = (q - self.q0).abs()
        r_dev = cf.w_dev_hip * (dev * self.m_hip).sum(-1) + cf.w_dev_arms * (dev * self.m_arm).sum(-1)
        r_arm_vel = (self._sl(s2, self.iqd) ** 2 * self.m_arm).sum(-1)

        f1, z1 = self.feet(s)
        f2, z2 = self.feet(s2)
        fv = v.unsqueeze(1) + torch.cross(w.unsqueeze(1).expand_as(f2), f2, dim=-1) + (f2 - f1) / self.dt
        fv_h = fv - (fv * g.unsqueeze(1)).sum(-1, keepdim=True) * g.unsqueeze(1)
        vh2 = (fv_h ** 2).sum(-1)
        c = torch.sigmoid(-z2 / cf.contact_band)                # 1 = foot on the ground
        r_clear = (torch.relu(cf.clearance_target - z2) * (1 - c) * vh2.clamp(min=1e-8).sqrt()).sum(-1) * active
        r_slip = (vh2 * c).sum(-1) * active
        cl, cr = c[:, 0], c[:, 1]
        # Locked to the clock, the mirror-symmetric actor's two half-cycles are mirror images: no one-sided limp.
        sp = self._sl(s2, self.iph)[:, 0]
        r_imit_q = r_imit_h = r_swing = torch.zeros_like(sp)
        if self.ref is None:
            want_r, want_l = (sp + 1) * 0.5, (1 - sp) * 0.5
            r_gait = walking * (1 - 0.5 * ((want_r - cr).abs() + (want_l - cl).abs()))
        else:                       # the table at the post-step (command, phase), as the env's imitation terms
            with torch.no_grad():
                ph = self._sl(s2, self.iph)
                phase = torch.atan2(ph[:, 0], ph[:, 1]) / (2 * math.pi) % 1.0
                standing = (cmd[:, :2].norm(dim=-1) < self.ref_stand) & (cmd[:, 2].abs() < self.ref_stand)
                ref = self.ref.sample(cmd, phase, standing)
                q_ref, h_ref, c_ref = self.ref.joint_field(ref, "joint_pos"), self.ref.field(ref, "foot_height"), self.ref.field(ref, "contact")
            sd = self.ref_std
            r_imit_q = torch.exp(-((q[:, self.ref_legs] - q_ref) ** 2).mean(-1) / sd["joint_pos"] ** 2)
            r_imit_h = torch.exp(-((z2.clamp(min=0) - h_ref) / sd["foot_height"]) ** 2).mean(-1)   # table and z2: [left, right]
            r_gait = 1 - (c - c_ref).abs().mean(-1)
            r_gait = r_gait * cf.w_imit_c / cf.w_gait
            # contralateral arm swing from the table's hip pitch
            pol = [self.names[int(i)] for i in self.ref_legs]
            n_ = self.names
            sh = [n_.index("l_shoulder_pitch"), n_.index("r_shoulder_pitch")]
            hips = [n_.index("r_hip_pitch_joint"), n_.index("l_hip_pitch_joint")]             # opposite legs
            hip_ref = q_ref[:, [pol.index("r_hip_pitch_joint"), pol.index("l_hip_pitch_joint")]]
            sw_ref = self.q0[sh] + cf.arm_swing_gain * (hip_ref - self.q0[hips])
            r_swing = walking * torch.exp(-((q[:, sh] - sw_ref) ** 2).mean(-1) / cf.std_arm_swing ** 2)
        r_still = stand * (1 - c).sum(-1)
        roll = self._roll
        r_flat2, r_flat1 = (roll ** 2 * c).sum(-1), ((roll ** 2 + 1e-6).sqrt() * c).sum(-1)

        t = {"lin": cf.w_lin * r_lin, "ang": cf.w_ang * r_ang, "upright": cf.w_upright * r_up, "pose": cf.w_pose * r_pose,
             "limits": cf.w_limits * r_lim, "action_rate": cf.w_action_rate * r_rate, "lin_z": cf.w_lin_z * r_lin_z,
             "ang_xy": cf.w_ang_xy * r_ang_xy, "clearance": cf.w_clearance * r_clear, "slip": cf.w_slip * r_slip,
             "gait": cf.w_gait * r_gait, "imit_q": cf.w_imit_q * r_imit_q, "imit_h": cf.w_imit_h * r_imit_h, "still": cf.w_still * r_still, "dev": r_dev, "arm_vel": cf.w_arm_vel * r_arm_vel, "arm_swing": cf.w_arm_swing * r_swing,
             "flat": cf.w_flat_l2 * r_flat2 + cf.w_flat_l1 * r_flat1}
        if terms:
            return {k: v * self.dt for k, v in t.items()}
        return sum(t.values()) * self.dt

    def term_logit(self, s2: torch.Tensor) -> torch.Tensor:
        """Logit of the tilt termination: tilt beyond the limit <=> -g_z < cos(limit)."""
        cos_tilt = -self._sl(s2, self.ig)[:, 2]
        return (math.cos(self.cfg.tilt_limit) - cos_tilt) / self.cfg.tilt_softness


@dataclass
class GetUpCfg:
    """Get-up terms, the analytic counterparts of envs/isaaclab/rewards/getup_rewards.py (config/chhaya_getup.yaml
    weights). No tilt termination: lying down is where every episode starts."""
    w_upright: float = 1.0          # ((up + 1) / 2)^2
    w_height: float = 2.0           # h_hat * (up + 1) / 2
    w_feet: float = 0.5             # [mostly upright] * mean(feet on the ground)
    w_stand: float = 3.0            # stand score * pose quality
    w_feet_flat: float = 1.5        # stand score * flat soles on the ground
    w_still: float = -1.0           # stand score * base speed^2
    # Smoothness (the first run thrashed): action changes, joint speed, and a soft cap on joint speed. A servo
    # jerking at 5 rad/s through its gears is what breaks a printed robot in a fall.
    w_action_rate: float = -0.5
    w_joint_vel: float = -0.005    # sum qd^2 over the 20 joints
    w_joint_vel_cap: float = -0.2  # sum relu(|qd| - joint_vel_cap)^2
    joint_vel_cap: float = 3.0
    w_limits: float = -1.0
    stand_height_frac: float = 0.9
    stand_tilt_deg: float = 15.0
    feet_min_up: float = 0.7
    std_pose: float = 0.3
    std_roll: float = 0.05
    contact_band: float = 0.004


class GetUpMDP(ChhayaMDP):
    """ChhayaMDP's state, integration and foot kinematics with the get-up reward and no termination."""

    def __init__(self, ph: dict, device, h_floor: float, h_stand: float, cfg: GetUpCfg = GetUpCfg()):
        super().__init__(ph, device)
        self.gcfg, self.h_floor, self.h_stand = cfg, float(h_floor), float(h_stand)
        self.max_step_reward = self.dt * (cfg.w_upright + cfg.w_height + cfg.w_feet + cfg.w_stand + cfg.w_feet_flat)

    def reward(self, s: torch.Tensor, a: torch.Tensor, s2: torch.Tensor, terms: bool = False):
        cf = self.gcfg
        v, w, g = self._sl(s2, self.iv), self._sl(s2, self.iw), self._sl(s2, self.ig)
        q, h = self._sl(s2, self.iq) + self.q0, self._sl(s2, self.ih)[:, 0]
        a_prev = torch.zeros_like(a)
        a_prev[:, self.asrc] = s[:, self.adst]
        up = -g[:, 2]
        h_hat = ((h - self.h_floor) / (self.h_stand - self.h_floor)).clamp(-0.5, 1.2)
        _, z2 = self.feet(s2)
        c = torch.sigmoid(-z2 / cf.contact_band)                       # 1 = foot on the ground
        score = (torch.sigmoid((h_hat - cf.stand_height_frac) / 0.03)
                 * torch.sigmoid((up - math.cos(math.radians(cf.stand_tilt_deg))) / 0.01))
        pose = torch.exp(-(((q - self.q0) / cf.std_pose) ** 2).mean(-1))
        t = {"upright": cf.w_upright * ((up + 1) / 2) ** 2,
             "height": cf.w_height * h_hat.clamp(0.0, 1.0) * (up + 1) / 2,
             "feet": cf.w_feet * torch.sigmoid((up - cf.feet_min_up) / 0.05) * c.mean(-1),
             "stand": cf.w_stand * score * pose,
             "feet_flat": cf.w_feet_flat * score * (c * torch.exp(-self._roll ** 2 / cf.std_roll ** 2)).mean(-1),
             "still": cf.w_still * score * ((v ** 2).sum(-1) + (w ** 2).sum(-1)),
             "action_rate": cf.w_action_rate * ((a.clamp(-1, 1) - a_prev) ** 2).sum(-1),
             "joint_vel": cf.w_joint_vel * (self._sl(s2, self.iqd) ** 2).sum(-1),
             "joint_vel_cap": cf.w_joint_vel_cap * (torch.relu(self._sl(s2, self.iqd).abs() - cf.joint_vel_cap) ** 2).sum(-1),
             "limits": cf.w_limits * (torch.relu(q - self.hi) + torch.relu(self.lo - q)).sum(-1)}
        if terms:
            return {k: x * self.dt for k, x in t.items()}
        return sum(t.values()) * self.dt

    def term_logit(self, s2: torch.Tensor) -> torch.Tensor:
        return torch.full((s2.shape[0],), -20.0, device=s2.device)     # never terminates (time-outs only)


# Nominal errors (Bryson's rule) for Chhaya's physical state; everything else is scaled by its std.
NOMINAL = {"v_b": 0.02, "w_b": 0.2, "q": 0.02, "qd": 0.5, "h": 0.005, "g_b": 0.02}


class WorldModelEth(WorldModel):
    """Dynamics only; same call signature as WorldModel: (s, a, c) -> (next state, reward, termination logit)."""

    def __init__(self, state_dim: int, action_dim: int, ctx_dim: int, layout: Layout, hidden, mdp: ChhayaMDP):
        super().__init__(state_dim, action_dim, ctx_dim, layout, hidden=(8,), head_hidden=(8,))
        del self.rew, self.term                                   # ponytail: reuse v1's normalisers, drop its heads
        self.dyn = mlp_ln(state_dim + action_dim + ctx_dim, hidden, state_dim)
        self.mdp = mdp
        nominal = torch.zeros(state_dim)
        for key, val in NOMINAL.items():
            lo, hi = layout.phys[key]
            nominal[lo:hi] = val
        self.register_buffer("nominal", nominal)

    def value_bounds(self, gamma: float):
        hi = self.mdp.max_step_reward / (1.0 - gamma)
        return -hi, hi                                            # ponytail: penalties assumed no larger than the bonuses

    def next_state(self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ns, nc = self.norm_s(s), self.norm_c(c)
        s2 = s + self.dyn(torch.cat([ns, a, nc], -1)) * self.d_std
        s2 = torch.where(self.analytic, self._analytic(s, a), s2)
        return self.mdp.integrate(s, s2)

    def forward(self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor):
        s2 = self.next_state(s, a, c)
        return s2, self.mdp.reward(s, a, s2), self.mdp.term_logit(s2)

    def window_loss(self, s, a, r, term, done, c):
        """Eq. 1 with analytic reward and termination heads: the state term only, on k-step windows."""
        B, k = a.shape[:2]
        scale = torch.where(self.nominal > 0, self.nominal, self.s_std)
        alive = torch.ones(B, device=s.device)
        x, ls, n = s[:, 0], torch.zeros((), device=s.device), torch.zeros((), device=s.device)
        for j in range(k):
            x = self.next_state(x, a[:, j], c)          # the fit needs no reward (no FK)
            m = alive * (1.0 - done[:, j])
            ls = ls + (((x - s[:, j + 1]) / scale).square().mean(-1) * m).sum()
            n = n + m.sum()
            alive = alive * (1.0 - done[:, j])
            x = torch.where((alive > 0).unsqueeze(-1), x, s[:, j + 1])
        ls = ls / n.clamp(min=1)
        return ls, {"wm/state": ls.detach()}
