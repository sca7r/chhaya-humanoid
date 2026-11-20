"""BipedGetUpEnv: the get-up / fall-recovery task for Chhaya (--task ChhayaGetUp).

A subclass of the walking BipedEnv that keeps everything sim2real about it (BAM servo model, backlash, the full sensor
model, DR and its curriculum, the 50-dim observation frame, the residual action map, mirror symmetry) and replaces
only the task: where episodes start, what is rewarded, when they end.

Hardware protocol it is trained for (the user's call, 2026-11-01): the walking policy faults at 45 deg tilt and turns
the servo torque OFF; the robot falls limp and the Jetson waits until it lies still (settle criterion), re-arms the
servos at the CURRENT pose and runs this policy; once the success criterion holds it hands back to walking.

  * Starts: getup.start_mix of a fall bank (utils/fall_bank.py, made by src/isaaclab/make_fall_bank.py with exactly
    that torque-off fall + settle) and the walking reset (standing at home: stay up, stay still).
    Re-arm convention (start_protocol in the bundle): history zero-filled, EMA state = the pose, previous action =
    the action of the pose (targets_to_actions). The IMU lag buffer is seeded with the real reading of the start pose
    (the walking env zero-fills it, which would show a lying robot gravity = 0 for a step).
  * Observation: the walking 50-dim frame with the command = 0 and the gait phase = (0, 0). Zeros, not (sin 0, cos 1):
    the mirror negates the phase dims, so cos = 1 would make every mirrored sample differ from every real one.
  * Critic: the walking 60 dims + 2 mirror-invariant extras (assist force fraction, base contact force / m g).
  * Rewards: envs/isaaclab/rewards/getup_rewards.py (staged task terms + hardware-safety penalties) plus the reused
    smoothness / limit terms. No tilt or contact terminations; time-out only (plus a NaN / blow-up guard).
  * Success (TensorBoard, the assist curriculum) = standing (height, tilt, both feet) AND the bundle's hand-off test
    (gyro, every joint within handoff.joint_err_max_rad of the home), held success_hold_s: exactly what the Jetson
    switches to walking on. (Until 2026-11-02 it ignored the pose: run 2026-11-01_18-59-35 logged 98% success in a
    stance the hand-off test accepted 0% of the time in MuJoCo.)
  * Success is the GOAL (getup.terminate_on_success, training only, 2026-11-04): the episode terminates there with
    the reward getup_success = the value of standing at the home pose forever (the per-step maximum x gamma /
    (1 - gamma)), so the Q scale is the one a perfect hold would earn, without the policy having to hold. After the
    hand-off, holding and push recovery are the WALKING policy's job. Runs 2026-11-01_18-59-35 and
    2026-11-04_07-22-45 both got up and then, over the ~7 s of holding per episode, drifted into a wide, low,
    passively stable A-frame (soles 200-250 mm apart, near-zero torque), even from standing starts at home, although
    holding home open loop in this env (full DR, pushes) scores 7.8/step vs the A-frame's 5.4 and falls less (0.7%
    vs 2%): the learned value of a closed-loop narrow hold, under noisy delayed sensing, exploration noise and
    pushes, lost to a stance whose value is trivially safe.
  * Assist-force curriculum (HoST): see getup_env_cfg.GETUP_DEFAULTS["assist"]. Training only. It decays on the
    RISE rate (standing: height, tilt, both feet, held success_hold_s), as HoST decays on reaching the target height,
    not on the strict pose-aware success: early policies cannot hold the home pose yet, and an assist that never
    decays props up a low crouch (it only pushes below assist.height_frac).
"""
from __future__ import annotations

import math

import torch

from . import quat as math_utils
from .compat import write_joint_state, write_root_pose, write_root_velocity

from . import rewards
from .biped_env import BipedEnv
from .getup_env_cfg import ChhayaGetUpEnvCfg
from .rewards import getup_rewards as GR

# Joint types in policy order pairs (l, r), for the per-type joint-at-limit diagnostics.
_JOINT_TYPES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll", "ankle_pitch")
_AT_LIMIT_RAD = 0.02

# The reward terms _get_rewards implements (env_config.reward_weights keys). A nonzero weight on anything else raises:
# a walking term copied into the get-up config would otherwise be silently ignored.
GETUP_REWARD_TERMS = ("getup_upright", "getup_height", "getup_feet_support", "getup_stand", "getup_feet_flat",
                      "getup_still", "getup_success",
                      "servo_overload", "body_impact", "ang_vel_limit", "self_contact_l2",
                      "action_rate_l2", "dof_torques_l2", "dof_acc_l2", "dof_pos_limits")


class BipedGetUpEnv(BipedEnv):
    cfg: ChhayaGetUpEnvCfg

    EXTRA_CRITIC_DIM = 2
    LOG_WALKING_DIAGNOSTICS = False

    def __init__(self, cfg: ChhayaGetUpEnvCfg, render_mode: str | None = None, **kwargs):
        g = cfg.getup
        if cfg.reference_npz_path:
            raise ValueError("the get-up task takes no gait table (remove env_config.reference)")
        unknown = sorted(k for k, v in cfg.rewards.items() if float(v) != 0.0 and k not in GETUP_REWARD_TERMS)
        if unknown:
            raise ValueError(f"get-up env: reward_weights {unknown} are not get-up terms (implemented: "
                             f"{list(GETUP_REWARD_TERMS)})")
        mix = {k: float(v) for k, v in dict(g["start_mix"]).items()}
        if set(mix) - {"bank", "standing"} or min(mix.values(), default=-1.0) < 0.0 or sum(mix.values()) <= 0.0:
            raise ValueError(f"getup.start_mix {mix}: keys bank / standing, non-negative, not all zero")
        if not cfg.scene.contact_forces.filter_shape_prim_expr:
            raise ValueError("the get-up env needs the ground-filtered contact sensor (flat terrain)")
        super().__init__(cfg, render_mode, **kwargs)
        self._g = g
        self._p_bank = mix.get("bank", 0.0) / sum(mix.values())
        dt = self.step_dt

        # ---- fall bank (device tensors) -----------------------------------------------------------------------
        self.bank_t = None
        bank_meta = None
        if self._p_bank > 0.0:
            import os
            from utils.fall_bank import CATEGORIES, load_bank
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            bank = load_bank(g["bank_path"], root_dir=root)
            names = bank.meta.get("joint_names")
            if names is not None and list(names) != list(self.robot.joint_names):
                raise ValueError(f"fall bank joint order {names} != the robot's {list(self.robot.joint_names)}")
            idx = bank.select(split=g["bank_split"], category=g.get("start_category"))
            if len(idx) == 0:
                raise ValueError(f"fall bank {g['bank_path']}: no states for split={g['bank_split']!r} "
                                 f"category={g.get('start_category')!r}")
            a = bank.arrays
            f = lambda k: torch.as_tensor(a[k][idx], dtype=torch.float32, device=self.device)
            self.bank_t = {k: f(k) for k in ("root_z", "root_quat", "joint_pos", "joint_vel",
                                              "root_lin_vel_b", "root_ang_vel_b")}
            self.bank_t["category"] = torch.as_tensor(a["category"][idx], dtype=torch.long, device=self.device)
            self._bank_n = len(idx)
            self._categories = CATEGORIES
            bank_meta = bank.meta
            print(f"[getup] fall bank {g['bank_path']}: {self._bank_n} {g['bank_split']} states "
                  f"{bank.histogram(idx)}; start mix bank {self._p_bank:.2f}")
        else:
            from utils.fall_bank import CATEGORIES
            self._categories = CATEGORIES

        # ---- heights for h_hat ------------------------------------------------------------------------------
        self.h_stand = float(getattr(cfg, "spawn_height_m", None) or self.rd.default_root_pose[0, 2].item())
        h_floor = g.get("h_floor_m")
        if h_floor is None:
            if bank_meta is None:
                raise ValueError("getup.h_floor_m is required when no fall bank is used (start_mix bank 0)")
            h_floor = float(bank_meta["h_floor_m"])
        self.h_floor = float(h_floor)
        if not (0.0 < self.h_floor < self.h_stand):
            raise ValueError(f"getup heights: need 0 < h_floor ({self.h_floor}) < h_stand ({self.h_stand})")
        print(f"[getup] h_floor {self.h_floor:.4f} m, h_stand {self.h_stand:.4f} m")

        self._weight_n = float(self._default_mass_dev[0].sum().item()) * 9.81
        self._base_robot_idx = self.robot.find_bodies(self.cfg.base_body_name)[0][0]
        self._base_cs_idx = int(self._base_id[0])
        # leg bodies + feet in contact-sensor order: the self-contact penalty (floor part removed)
        self._self_contact_ids_t = torch.as_tensor(sorted(set(self._leg_body_ids) | set(self._feet_ids)),
                                                   device=self.device, dtype=torch.long)
        self._home = self.default_joint_pos
        self._hold_steps = max(1, int(math.ceil(float(g["success_hold_s"]) / dt - 1e-6)))

        # ---- per-env episode trackers -----------------------------------------------------------------------
        n, j = self.num_envs, self.num_joints
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.overload_state = z(n, j)
        self.stand_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=self.device)
        # rise = standing (height, tilt, feet) held success_hold_s, pose not required: drives the assist curriculum
        self.rise_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.rose = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.t_stand = torch.full((n,), float("nan"), device=self.device)
        self.start_cat = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.peak_ang_vel, self.peak_base_force, self.overload_peak = z(n), z(n), z(n)
        self.at_limit_steps = z(n, j)
        # posture while standing (the hard stand mask): per-joint squared error to the home, foot roll, steps
        self.pose_err2_sum, self.foot_roll_sum, self.posture_steps = z(n, j), z(n), z(n)
        self.handoff_steps, self.h_hat_sum = z(n), z(n)
        self.assist_applied = z(n)
        self._base_force_now = z(n)
        self.standing_now = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._newly_succeeded = torch.zeros(n, dtype=torch.bool, device=self.device)

        # ---- assist curriculum ------------------------------------------------------------------------------
        asc = g["assist"]
        self.assist_frac = float(asc["start_frac"]) if asc["enabled"] else 0.0
        self._assist_wrench_on = False
        self._win_eps = z(())
        self._win_succ = z(())
        self._assist_next_update = int(asc["update_steps"])
        self._assist_last_rate = float("nan")
        self._assist_forces = z(n, 1, 3)
        self._assist_torques = z(n, 1, 3)

    # ------------------------------------------------------------------ observation hooks
    def _phase_obs(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 2, device=self.device)

    def _extra_critic_obs(self) -> list[torch.Tensor]:
        return [self.assist_applied.unsqueeze(-1), (self._base_force_now / self._weight_n).unsqueeze(-1)]

    # ------------------------------------------------------------------ actions / assist
    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        self.commands.zero_()              # no velocity command in this task (the generator may still resample)
        self._apply_assist()

    def _apply_assist(self) -> None:
        """Upward k * m g on base_link while roughly upright and still low; written once per control step (the
        articulation applies the buffered wrench every physics substep)."""
        if self.assist_frac <= 0.0:
            if self._assist_wrench_on:
                self._assist_forces.zero_()
                self.robot.set_external_force_and_torque(self._assist_forces, self._assist_torques,
                                                         body_ids=[self._base_robot_idx], is_global=True)
                self._assist_wrench_on = False
            self.assist_applied.zero_()
            return
        asc = self._g["assist"]
        d = self.rd
        up = GR.uprightness(d.projected_gravity_b)
        h_hat = GR.height_progress(d.root_pos_w[:, 2] - self.scene.env_origins[:, 2], self.h_floor, self.h_stand)
        active = (up > float(asc["min_up"])) & (h_hat < float(asc["height_frac"]))
        self.assist_applied = active.float() * self.assist_frac
        self._assist_forces[:, 0, 2] = self.assist_applied * self._weight_n
        self.robot.set_external_force_and_torque(self._assist_forces, self._assist_torques,
                                                 body_ids=[self._base_robot_idx], is_global=True)
        self._assist_wrench_on = True

    # ------------------------------------------------------------------ rewards
    def _getup_state(self) -> dict:
        """Ground-truth quantities shared by the rewards, the success test and the diagnostics (fresh robot.data)."""
        d, cs, g = self.rd, self.cs, self._g
        grav = d.projected_gravity_b
        up = GR.uprightness(grav)
        h_hat = GR.height_progress(d.root_pos_w[:, 2] - self.scene.env_origins[:, 2], self.h_floor, self.h_stand)
        feet_loaded = torch.norm(cs.net_forces_w[:, self._feet_ids_t], dim=-1) > 1.0
        h_hi, tilt = float(g["stand_height_frac"]), float(g["stand_tilt_deg"])
        standing = GR.standing_mask(h_hat, up, feet_loaded, h_hi, tilt)
        up_hi = math.cos(math.radians(tilt))
        score = GR.stand_score(h_hat, up, feet_loaded, float(g["stand_soft_height_frac"]), h_hi,
                               math.cos(math.radians(float(g["stand_soft_tilt_deg"]))), up_hi)
        # the Jetson's hand-off test (tilt is already in `standing`: stand_tilt_deg)
        ho = g["handoff"]
        err_max = (d.joint_pos - self._home).abs().max(dim=-1)[0]
        handoff_ok = (standing & (up > math.cos(math.radians(float(ho["tilt_max_deg"]))))
                      & (torch.norm(d.root_ang_vel_b, dim=-1) < float(ho["gyro_max_rads"]))
                      & (err_max < float(ho["joint_err_max_rad"])))
        base_force = torch.norm(cs.net_forces_w_history[:, :, self._base_cs_idx], dim=-1).max(dim=1)[0]
        ids = self._self_contact_ids_t
        net = cs.net_forces_w_history[:, :, ids]                                    # [N, T, K, 3]
        gnd = cs.force_matrix_w_history[:, :, ids].sum(dim=3)                       # floor part only
        self_f = torch.norm(net - gnd, dim=-1).max(dim=1)[0]                         # [N, K]
        return {"grav": grav, "up": up, "h_hat": h_hat, "feet_loaded": feet_loaded, "standing": standing,
                "stand_score": score, "handoff_ok": handoff_ok,
                "base_force": base_force, "self_force": self_f}

    def _get_rewards(self) -> torch.Tensor:
        d, g = self.rd, self._g
        s = self._s                           # computed by _get_dones this step (DirectRLEnv.step runs it first)
        q, qd, tau = d.joint_pos, d.joint_vel, d.applied_torque
        feet_quat = d.body_quat_w[:, self._feet_idx_t]
        self.overload_state = GR.overload_step(self.overload_state, tau, float(g["stall_torque_nm"]),
                                               float(g["overload_threshold_frac"]), float(g["overload_leak_s"]),
                                               self.step_dt)
        w = lambda k: float(self.cfg.rewards.get(k, 0.0))
        pose_q = GR.pose_quality(q, self._home, float(g["pose_sharp_rad2"]), float(g["pose_broad_rad2"]))
        roll_l1 = rewards.foot_flat_roll_l1(feet_quat, d.root_quat_w, s["feet_loaded"])
        flat_q = torch.exp(-roll_l1 / float(g["foot_flat_sigma"]))      # = GR.foot_flat_quality, roll_l1 reused below
        raw = {   # keys == GETUP_REWARD_TERMS
            "getup_upright": GR.getup_upright(s["grav"]),
            "getup_height": GR.getup_height(s["h_hat"], s["grav"]),
            "getup_feet_support": GR.getup_feet_support(s["grav"], s["feet_loaded"], float(g["feet_support_min_up"])),
            "getup_stand": GR.getup_stand(s["stand_score"], pose_q),
            "getup_feet_flat": GR.getup_feet_flat(s["stand_score"], flat_q),
            "getup_still": GR.getup_still(s["stand_score"] * pose_q, d.root_lin_vel_b, d.root_ang_vel_b, qd),
            "getup_success": self._newly_succeeded.float(),
            "servo_overload": GR.servo_overload(self.overload_state),
            "body_impact": GR.body_impact(s["base_force"], self._weight_n, float(g["impact_weight_mult"])),
            "ang_vel_limit": GR.ang_vel_limit(d.root_ang_vel_b, float(g["ang_vel_limit_rads"])),
            "self_contact_l2": GR.self_contact_l2(s["self_force"], float(g["self_contact_thresh_n"]),
                                                  float(g["self_contact_sat_n"])),
            "action_rate_l2": rewards.action_rate_l2(self.actions, self.previous_actions),
            "dof_torques_l2": rewards.joint_torques_l2(tau),
            "dof_acc_l2": torch.sum(torch.square(d.joint_acc), dim=1),
            "dof_pos_limits": rewards.joint_pos_limits(q, self.joint_limits_min, self.joint_limits_max),
        }
        total = self._sum_reward_terms({k: v * w(k) for k, v in raw.items()})

        # diagnostics
        self.peak_ang_vel = torch.maximum(self.peak_ang_vel, torch.norm(d.root_ang_vel_b, dim=-1))
        self.peak_base_force = torch.maximum(self.peak_base_force, s["base_force"])
        self.overload_peak = torch.maximum(self.overload_peak, self.overload_state.max(dim=-1)[0])
        self.at_limit_steps += (((q - self.joint_limits_min) < _AT_LIMIT_RAD)
                                | ((self.joint_limits_max - q) < _AT_LIMIT_RAD)).float()
        up_high = s["standing"].float()
        self.pose_err2_sum += up_high.unsqueeze(-1) * torch.square(q - self._home)
        n_loaded = s["feet_loaded"].float().sum(dim=-1)
        roll = torch.asin(torch.clamp(roll_l1 / n_loaded.clamp(min=1.0), 0.0, 1.0))
        self.foot_roll_sum += up_high * (n_loaded > 0).float() * roll
        self.posture_steps += up_high
        self.handoff_steps += s["handoff_ok"].float()
        self.h_hat_sum += s["h_hat"]
        self._base_force_now = s["base_force"]
        self._accumulate_torque_stats()

        # a_t into the next observation (same convention as the walking env)
        self.previous_previous_actions = self.previous_actions.clone()
        self.previous_actions = self.actions.clone()
        return total

    # ------------------------------------------------------------------ terminations
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Success (getup.terminate_on_success) terminates; otherwise time-out only. NaN or a blown-up state (base
        > 2 m from the floor or > 20 m/s) ends the episode so it never reaches the replay buffer; falling over is the
        task, not a failure. Runs before _get_rewards: the success bookkeeping lives here and _get_rewards pays
        getup_success on exactly the terminating step."""
        s = self._s = self._getup_state()
        # success: the hand-off test (standing + still + every joint near the home) held for success_hold_s, sticky per
        # episode; time-to-stand = when the hold began
        standing = s["handoff_ok"]
        self.stand_count = torch.where(standing, self.stand_count + 1, torch.zeros_like(self.stand_count))
        newly = (self.stand_count >= self._hold_steps) & ~self.succeeded
        t0 = (self.episode_length_buf - self._hold_steps).clamp(min=0).float() * self.step_dt
        self.t_stand = torch.where(newly, t0, self.t_stand)
        self.succeeded |= newly
        self._newly_succeeded = newly
        self.standing_now = standing
        self.rise_count = torch.where(s["standing"], self.rise_count + 1, torch.zeros_like(self.rise_count))
        self.rose |= self.rise_count >= self._hold_steps

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        d = self.rd
        bad = (torch.isnan(d.joint_pos).any(dim=-1) | torch.isnan(d.joint_vel).any(dim=-1)
               | torch.isnan(d.root_pos_w).any(dim=-1)
               | ((d.root_pos_w[:, 2] - self.scene.env_origins[:, 2]).abs() > 2.0)
               | (torch.norm(d.root_lin_vel_w, dim=-1) > 20.0))
        if self._g.get("terminate_on_success", False):
            return bad | newly, time_out & ~newly
        return bad, time_out

    # ------------------------------------------------------------------ resets
    def _reset_idx(self, env_ids: torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        logs = self._episode_logs(env_ids)          # before super() and the tracker reset below
        self._update_assist(env_ids)
        super()._reset_idx(env_ids)
        self.commands[env_ids] = 0.0
        for act in self.bam_actuators:              # always powered in training (make_fall_bank turns them off itself)
            act.torque_enable[env_ids] = 1.0
        self.start_cat[env_ids] = -1
        # Standing starts follow the same re-arm convention as the bank starts (previous action = the action of the
        # pose, not the walking reset's 0): the Jetson starts this policy that way whatever pose it re-arms in.
        d = self.rd
        self._reseed_after_joint_write(env_ids, d.joint_pos[env_ids].clone(), d.joint_vel[env_ids].clone())
        if self.bank_t is not None:
            pick = env_ids[torch.rand(len(env_ids), device=self.device) < self._p_bank]
            if len(pick) > 0:
                bi = torch.randint(0, self._bank_n, (len(pick),), device=self.device)
                b = self.bank_t
                self.write_start_states(pick, b["root_z"][bi], b["root_quat"][bi], b["joint_pos"][bi],
                                        b["joint_vel"][bi], b["root_lin_vel_b"][bi], b["root_ang_vel_b"][bi])
                self.start_cat[pick] = b["category"][bi]
        self._seed_imu_buffer(env_ids)
        for buf in (self.overload_state, self.peak_ang_vel, self.peak_base_force, self.overload_peak,
                    self.at_limit_steps, self.assist_applied, self._base_force_now,
                    self.pose_err2_sum, self.foot_roll_sum, self.posture_steps, self.handoff_steps, self.h_hat_sum):
            buf[env_ids] = 0.0
        self.stand_count[env_ids] = 0
        self.succeeded[env_ids] = False
        self.rise_count[env_ids] = 0
        self.rose[env_ids] = False
        self.standing_now[env_ids] = False
        self.t_stand[env_ids] = float("nan")
        self.extras["log"].update(logs)
        self.extras["log"]["Curriculum/assist_force_frac"] = self.assist_frac
        if not math.isnan(self._assist_last_rate):
            self.extras["log"]["Curriculum/assist_window_rise_rate"] = self._assist_last_rate

    def write_start_states(self, ids: torch.Tensor, root_z: torch.Tensor, quat_noyaw: torch.Tensor,
                           q: torch.Tensor, qd: torch.Tensor, lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor,
                           yaw: torch.Tensor | None = None) -> None:
        """Place envs `ids` in a given state: base height above the floor, tilt (yaw-free quat) composed with a heading
        (default: the heading the reset event just drew), joints, base-frame velocities; then re-seed the buffers that
        _reset_idx derived from the old pose (filtered targets, delays, previous action = the re-arm convention)."""
        d = self.rd
        if yaw is None:
            yaw = math_utils.euler_xyz_from_quat(d.root_quat_w[ids])[2]
        zeros = torch.zeros_like(yaw)
        quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), quat_noyaw)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        write_joint_state(self.robot, q, qd, env_ids=ids)
        pos = d.root_pos_w[ids].clone()
        pos[:, 2] = self.scene.env_origins[ids, 2] + root_z + float(self._g["spawn_clearance_m"])
        write_root_pose(self.robot, torch.cat([pos, quat], dim=-1), env_ids=ids)
        vel = torch.cat([math_utils.quat_apply(quat, lin_vel_b), math_utils.quat_apply(quat, ang_vel_b)], dim=-1)
        write_root_velocity(self.robot, vel, env_ids=ids)
        self._reseed_after_joint_write(ids, q, qd)

    def _seed_imu_buffer(self, env_ids: torch.Tensor) -> None:
        """Every slot of the IMU lag buffer = the reading at the start pose: specific force -g * grav_b (at rest),
        gyro, gravity direction. (Base frame; the env's misalignment rotation and biases apply on read-out.)"""
        if self.max_imu_obs_delay <= 0:
            return
        d = self.rd
        quat = d.root_quat_w[env_ids]
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(len(env_ids), 3)
        grav_b = math_utils.quat_apply_inverse(quat, down)
        ang_b = math_utils.quat_apply_inverse(quat, d.root_ang_vel_w[env_ids])
        frame = torch.cat([-9.81 * grav_b, ang_b, grav_b], dim=-1)
        self.imu_obs_buffer[env_ids] = frame.unsqueeze(1).expand(-1, self.imu_obs_buffer.shape[1], -1)

    # ------------------------------------------------------------------ curriculum + logging
    def _update_assist(self, env_ids: torch.Tensor) -> None:
        """Window counts of finished fall-bank episodes and how many ROSE (see the module docstring); every update_steps
        control steps one host sync decides whether k drops by step_frac."""
        asc = self._g["assist"]
        valid = (self.episode_length_buf[env_ids] > 0) & (self.start_cat[env_ids] >= 0)
        self._win_eps += valid.sum()
        self._win_succ += (valid & self.rose[env_ids]).sum()
        if not asc["enabled"] or self.assist_frac <= float(asc["min_frac"]):
            return
        if self.common_step_counter < self._assist_next_update:
            return
        self._assist_next_update = self.common_step_counter + int(asc["update_steps"])
        eps, succ = float(self._win_eps.item()), float(self._win_succ.item())
        if eps < float(asc["min_episodes"]):
            return
        self._assist_last_rate = succ / eps
        if self._assist_last_rate > float(asc["success_threshold"]):
            self.assist_frac = max(float(asc["min_frac"]), self.assist_frac - float(asc["step_frac"]))
            print(f"[getup] assist force -> {self.assist_frac:.2f} m g (window rise rate {self._assist_last_rate:.2f} "
                  f"over {int(eps)} episodes)")
        self._win_eps.zero_()
        self._win_succ.zero_()

    def _episode_logs(self, env_ids: torch.Tensor) -> dict:
        """Per-episode get-up numbers for the envs about to reset (device tensors; nan = no valid episode)."""
        done = self.episode_length_buf[env_ids] > 0               # the very first reset has no finished episode
        cat = self.start_cat[env_ids]
        succ = self.succeeded[env_ids].float()
        bank = (done & (cat >= 0)).float()
        stand = (done & (cat < 0)).float()
        rose = self.rose[env_ids].float()
        out = {"Diagnostics/getup_success_rate": (succ * bank).sum() / bank.sum(),
               "Diagnostics/getup_rise_rate": (rose * bank).sum() / bank.sum(),
               "Diagnostics/standing_start_success_rate": (succ * stand).sum() / stand.sum()}
        for i, name in enumerate(self._categories):
            m = (done & (cat == i)).float()
            out[f"Diagnostics/getup_success_{name}"] = (succ * m).sum() / m.sum()
        # terminations: the goal (hand-off reached) vs the NaN / blow-up guard (the base env would call both
        # base_contact; this dict is merged after its log, so these replace its count)
        term = self.reset_terminated[env_ids]
        goal = term & self._newly_succeeded[env_ids]
        out["Episode_Termination/success"] = torch.count_nonzero(goal).float()
        out["Episode_Termination/base_contact"] = torch.count_nonzero(term & ~goal).float()
        ok = (bank > 0) & self.succeeded[env_ids]
        t = torch.where(ok, self.t_stand[env_ids], torch.zeros_like(succ))
        out["Diagnostics/time_to_stand_s"] = t.sum() / ok.float().sum()
        dn = done.float()
        mean = lambda v: (v * dn).sum() / dn.sum()
        out["Diagnostics/servo_overload_peak"] = mean(self.overload_peak[env_ids])
        out["Diagnostics/peak_ang_vel_rads"] = mean(self.peak_ang_vel[env_ids])
        out["Diagnostics/base_impact_peak_n"] = mean(self.peak_base_force[env_ids])
        steps = self.episode_step_count[env_ids].clamp(min=1.0).unsqueeze(-1)
        frac = self.at_limit_steps[env_ids] / steps
        for i, t_name in enumerate(_JOINT_TYPES):
            out[f"Diagnostics/joint_at_limit_frac/{t_name}"] = mean(frac[:, 2 * i:2 * i + 2].mean(dim=-1))
        out["Diagnostics/handoff_ok_frac"] = mean(self.handoff_steps[env_ids] / steps[:, 0])
        # episode-mean base height (0 lying, 1 home stand): a policy parked low (run 2026-11-02_11-45-46 squatted at
        # ~0.55 for 233M steps) shows here long before the success rates say anything
        out["Diagnostics/mean_height_frac"] = mean(self.h_hat_sum[env_ids] / steps[:, 0])
        # posture while standing (height, tilt, both feet): RMS joint error to the home per type [deg], sole roll [deg]
        ps = self.posture_steps[env_ids]
        has = (done & (ps > 0)).float()
        rms = torch.rad2deg(torch.sqrt(self.pose_err2_sum[env_ids] / ps.clamp(min=1.0).unsqueeze(-1)))
        for i, t_name in enumerate(_JOINT_TYPES):
            out[f"Diagnostics/stand_pose_err_deg/{t_name}"] = (rms[:, 2 * i:2 * i + 2].mean(dim=-1) * has).sum() / has.sum()
        roll = torch.rad2deg(self.foot_roll_sum[env_ids] / ps.clamp(min=1.0))
        out["Diagnostics/stand_foot_roll_deg"] = (roll * has).sum() / has.sum()
        return out
