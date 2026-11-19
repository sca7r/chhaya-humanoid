"""Chhaya's Isaac Lab 3.0 environment (envs/isaaclab/biped_env.py) seen through the interface the
SuperMimic learner expects (supermimic/learner.py).

World-model state (noise-free, everything the next step depends on that we can read):
    [ proprio frame P (= the critic's copy of the policy observation, before sensor noise)
    | privileged: body-frame linear velocity 3, root height 1, foot forces in the body frame 6
    | true joint angles minus home (all joints) | true joint velocities
    | body-frame angular velocity 3 | gravity in the body frame 3
    | leg actions two steps back 12 (the action-delay pipeline can still be applying them) ]
The proprio frame carries the encoder offset, quantisation, delays and IMU bias the real robot sees;
the true joint and IMU values next to it make the dynamics close to Markov.

Sim context xi (per episode, Table A.3 of the paper, extended to our randomisation): per servo
group the mean firmware gain, supply voltage, friction scale, torque scale, dead band and gear
play; action delay; IMU delay and biases; per-joint encoder offsets; base mass and centre of mass;
mean contact friction. Values the env does not randomise in a run stay constant (harmless).
"""
import torch

from .world_model import Layout


class ChhayaAdapter:
    def __init__(self, env):
        self.env = env
        self.device = torch.device(env.device)
        self.n = env.num_envs
        self.action_dim = int(env.cfg.action_space)
        self._obs = None
        self.reset()
        P = env.obs_proprio_dim
        self.P = P
        nl = env.num_actions                       # leg actions (12)
        na = self.action_dim - nl                  # arm actions (8 for Chhaya, 0 legs-only)
        # proprio layout (biped_env._get_observations): acc 3, gyro 3, gravity 3, command 3, joint pos 12,
        # joint vel 12, previous leg actions 12, phase 2, [arm pos 8, arm vel 8, previous arm actions 8]
        prev_legs = list(range(36, 36 + nl))
        prev_arms = list(range(P - na, P)) if na else []
        cmd = list(range(9, 12))
        s = self.state()
        self.state_dim = s.shape[1]
        pp = list(range(self.state_dim - nl, self.state_dim))       # the "two steps back" slots, last in state
        self.ctx_dim = self.context().shape[1]
        self.layout = Layout(
            obs=list(range(P)), obs_noise_std=[float(x) for x in env.obs_noise_std.tolist()],
            action_dst=prev_legs + prev_arms, action_src=list(range(nl)) + list(range(nl, nl + na)),
            shift_dst=pp, shift_src=prev_legs, hold=cmd, history=max(int(env.history_size), 1), action_bound=1.0)
        if len(env.obs_noise_std) != P:
            raise ValueError("observation noise vector does not match the proprio frame")
        self.layout.phys = self._phys(P, nl, na)

    def _phys(self, P: int, nl: int, na: int) -> dict:
        """Where the physical quantities sit in state() (see its docstring), for supermimic/eth.py."""
        import os
        e, d = self.env, self.env.rd
        J = int(d.joint_pos.shape[1])
        lim = d.soft_joint_pos_limits[0]
        b = P + 10
        lay = self.layout
        return {"v_b": (P, P + 3), "h": (P + 3, P + 4), "q": (b, b + J), "qd": (b + J, b + 2 * J),
                "w_b": (b + 2 * J, b + 2 * J + 3), "g_b": (b + 2 * J + 3, b + 2 * J + 6), "cmd": (9, 12),
                "action_dst": list(lay.action_dst), "action_src": list(lay.action_src),
                "joint_names": list(e.robot.joint_names), "q0": d.default_joint_pos[0].tolist(),
                "soft_lo": lim[:, 0].tolist(), "soft_hi": lim[:, 1].tolist(),
                "dt": float(e.cfg.sim.dt * e.cfg.decimation), "decimation": int(e.cfg.decimation),
                "urdf": os.path.join(os.path.dirname(__file__), "..", "robot_hardware", "robot_description", "urdf",
                                     "Robot.urdf" if na == 0 else "chhaya.urdf")}

    def calibrate_sole(self, mdp) -> float:
        """Foot-link height when standing at reset = the rest height the reward's foot terms measure from."""
        mdp.sole = 0.0
        _, z = mdp.feet(self.state())
        mdp.sole = float(z.median())
        return mdp.sole

    def mirror_tables(self):
        """Left-right mirror of one proprio frame and of the action: (obs_perm, obs_sign, act_perm, act_sign)."""
        e = self.env
        return (e._mirror_obs_perm.long(), e._mirror_obs_signs.float(),
                e._mirror_action_perm.long(), e._mirror_action_signs.float())

    # ---------------------------------------------------------------- learner interface
    def reset(self) -> None:
        obs, _ = self.env.reset()
        self._obs = obs

    def policy_obs(self) -> torch.Tensor:
        return self._obs["policy"]

    def state(self) -> torch.Tensor:
        e, d = self.env, self.env.rd
        crit = self._obs["critic"]
        if crit.shape[1] < self.P + 10:
            raise ValueError("SuperMimic needs critic_has_privileged_info (the critic observation is its state)")
        return torch.cat([
            crit[:, :self.P + 10],                                   # proprio frame + standard privileged 10
            d.joint_pos - d.default_joint_pos, d.joint_vel,
            d.root_ang_vel_b, d.projected_gravity_b,
            e.previous_previous_actions,
        ], dim=-1).float()

    def context(self) -> torch.Tensor:
        e, n, dev = self.env, self.n, self.device
        parts = []
        for act in getattr(e, "bam_actuators", []):
            k = act.servo.kernel
            lm = act.servo.lost_motion
            parts += [k.kp.float().mean(-1, keepdim=True).expand(n, 1) if k.kp.dim() > 1 else k.kp.float().reshape(-1, 1).expand(n, 1),
                      k.vin.float().reshape(n, -1)[:, :1],
                      k.friction_scale.float().mean(-1, keepdim=True), k.torque_scale.float().mean(-1, keepdim=True),
                      k.deadband.float().mean(-1, keepdim=True),
                      lm.lash.float().mean(-1, keepdim=True) if lm is not None else torch.zeros(n, 1, device=dev)]
        parts += [e.action_delay.float().reshape(n, 1)]
        if hasattr(e, "imu_obs_delay"):
            parts.append(e.imu_obs_delay.float().reshape(n, 1))
        parts += [e.imu_accel_bias, e.imu_gyro_bias, e.mount_offset]
        d = e.rd
        try:
            parts += [d.body_mass[:, :1].float(), d.body_com_pos_b[:, 0].float()]
        except Exception:                          # field names differ between Isaac Lab versions
            parts.append(torch.zeros(n, 4, device=dev))
        parts.append(self._friction())
        return torch.cat([p.to(dev).float() for p in parts], dim=-1)

    def _friction(self) -> torch.Tensor:
        try:
            from isaaclab_newton.physics import NewtonManager
            mu = self.env.robot._root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0]
            from envs.isaaclab.compat import as_torch
            return as_torch(mu).float().reshape(self.n, -1).mean(-1, keepdim=True)
        except Exception:
            return torch.zeros(self.n, 1, device=self.device)

    def executed_action(self) -> torch.Tensor:
        """The action the env actually applied in the last step (after its own action noise, clamped)."""
        e = self.env
        parts = [e.actions] + ([e.arm_actions] if self.action_dim > e.num_actions else [])
        return torch.cat(parts, dim=-1).float()

    def step(self, action: torch.Tensor):
        obs, rew, term, trunc, _ = self.env.step(action)
        self._obs = obs
        return rew.float(), term.bool(), trunc.bool()

    # ---------------------------------------------------------------- checks
    @torch.no_grad()
    def check_layout(self, steps: int = 3) -> dict:
        """After a step the previous-action slots must equal the executed action, except in envs that
        just reset. Returns the largest mismatch per group."""
        lay, out = self.layout, {}
        for _ in range(steps):
            a = (torch.rand(self.n, self.action_dim, device=self.device) * 2.4 - 1.2)
            s0 = self.state()
            _, term, trunc = self.step(a)
            a = self.executed_action()
            s1 = self.state()
            ok = ~(term | trunc)
            if ok.any():
                dst, src = torch.tensor(lay.action_dst), torch.tensor(lay.action_src)
                out["prev_action"] = max(out.get("prev_action", 0.0),
                                         float((s1[ok][:, dst] - a[ok][:, src].clamp(-1, 1)).abs().max()))
                sd, ss = torch.tensor(lay.shift_dst), torch.tensor(lay.shift_src)
                out["shift"] = max(out.get("shift", 0.0), float((s1[ok][:, sd] - s0[ok][:, ss]).abs().max()))
        self.reset()            # the random actions above can knock robots over: start training from fresh episodes
        return out
