"""Config for the get-up / fall-recovery task (envs/isaaclab/getup_env.py, --task BipedRobotV2GetUp).

Everything the walking BipedRobotV2EnvCfg defines is inherited unchanged: the scene, the BAM servo model, backlash,
the sensor model (noise, vibration, delays, biases, quantisation), the DR events and their curriculum. The task-
specific settings live in one dict, `getup`, which config.yaml's env_config.getup block overrides key by key
(src/isaaclab/env_config._apply_getup; an unknown key raises). The training config is the source of truth: the
defaults below only document the intended values.
"""
from __future__ import annotations

import copy

from isaaclab.utils import configclass

from .biped_env_cfg import BipedRobotV2EnvCfg

GETUP_DEFAULTS: dict = {
    # ---- start states ------------------------------------------------------------------------------------------
    # Fall bank (src/isaaclab/make_fall_bank.py): settled limp falls, the state the Jetson re-arms from.
    "bank_path": "data_getup/fall_bank.npz",
    "bank_split": "train",               # train | gate (gate = the MuJoCo gate's held-out starts)
    # Fractions of resets: "bank" = a fall bank state, "standing" = the walking reset (home pose + small noise,
    # upright): it must stay up and still (false fall detections, and sitting falls end close to it).
    "start_mix": {"bank": 0.85, "standing": 0.15},
    "start_category": None,              # play / debugging: only bank states of this category (utils.fall_bank.CATEGORIES)
    "spawn_clearance_m": 0.002,          # bank states are placed this much higher than captured (no initial penetration)
    # ---- success / hand-off criterion (also exported for the Jetson) ---------------------------------------------
    "stand_height_frac": 0.9,            # h_hat above this ...
    "stand_tilt_deg": 15.0,              # ... base tilt below this ...
    "success_hold_s": 0.5,               # ... both feet loaded AND the handoff test below, held this long = success
    # Training only (env_config turns it off in play / eval): success TERMINATES the episode and pays getup_success
    # (its weight = the value of standing at home forever, see getup_env's docstring). The hand-off is the goal.
    "terminate_on_success": True,
    # Quality factors of the stand bonuses (getup_rewards.pose_quality / foot_flat_quality), all in [0, 1]:
    "pose_sharp_rad2": 0.25,             # pose: 0.5 exp(-E / sharp) + 0.5 exp(-E / broad), E = |q - home|^2 [rad^2]
    "pose_broad_rad2": 2.0,
    "foot_flat_sigma": 0.35,             # soles: exp(-sum |sin(roll)| / sigma) over loaded feet
    # Soft stand score (getup_stand / getup_feet_flat / getup_still): ramps from (stand_soft_height_frac, stand_soft_tilt_deg) up to
    # (stand_height_frac, stand_tilt_deg), times the loaded-feet fraction: no cliff at the threshold.
    "stand_soft_height_frac": 0.7,
    "stand_soft_tilt_deg": 30.0,
    "h_floor_m": None,                   # lying base height for h_hat; None = the bank's meta h_floor_m
    "feet_support_min_up": 0.7,          # getup_feet_support gate on up = -g_z
    # ---- penalty parameters ------------------------------------------------------------------------------------
    "overload_threshold_frac": 0.7,      # servo_overload: integrates |tau| / stall above this
    "overload_leak_s": 1.0,              # ... with this leak time constant
    "stall_torque_nm": 2.94,             # the capped stall torque (actuator.stall_torque_cap_nm)
    "impact_weight_mult": 2.0,           # body_impact: base contact force above this x m g
    "ang_vel_limit_rads": 4.0,           # ang_vel_limit: base rate above this
    "self_contact_thresh_n": 0.5,        # self_contact_l2 kernel (leg on leg, floor removed)
    "self_contact_sat_n": 5.0,
    # ---- assist-force curriculum (HoST): an upward force k * m g on base_link, only while the base is roughly
    # upright (up > min_up) and still low (h_hat < height_frac), so it can never hold up a standing robot. k drops by
    # step_frac whenever the RISE rate (standing held, pose not required; as HoST) of fall-bank episodes over the last
    # window exceeds success_threshold.
    # Training only: env_config turns it off in play / eval.
    "assist": {"enabled": True, "start_frac": 0.6, "step_frac": 0.05, "min_frac": 0.0,
               "success_threshold": 0.7, "update_steps": 250, "min_episodes": 256,
               "height_frac": 0.75, "min_up": 0.5},
    # ---- settle criterion: the Jetson re-arms once the robot lies still; the fall bank captures exactly then ----
    # gyro / joint speed below these for hold_s. lin_vel_ms (base speed; on the robot: |accel| within ~1 m/s^2 of g)
    # and min_s (time since the fault) keep a free-falling robot, whose gyro and joints are quiet, from counting.
    "settle": {"gyro_rads": 0.3, "joint_vel_rads": 0.5, "lin_vel_ms": 0.05, "hold_s": 0.3, "min_s": 0.5, "max_s": 4.0},
    # ---- hand-off to walking, in what the JETSON can measure (no foot sensors, no base height): the base within
    # tilt_max_deg of upright, |gyro| below gyro_max_rads, every joint within joint_err_max_rad of the walking home,
    # held hold_s. Exported in the bundle; the MuJoCo gate triggers the walking policy on exactly this test.
    "handoff": {"tilt_max_deg": 15.0, "gyro_max_rads": 0.5, "joint_err_max_rad": 0.2, "hold_s": 0.5},
    # ---- MuJoCo gate / hand-off (src/mujoco/getup_gate.py) --------------------------------------------------------
    "episode_s": 10.0,                   # episode length (also env episode_length_s)
}


@configclass
class BipedRobotV2GetUpEnvCfg(BipedRobotV2EnvCfg):
    # 10 s: a get-up takes 2-4 s, a failed attempt can be retried inside the episode, and the rest is holding still.
    episode_length_s = 10.0

    getup: dict = copy.deepcopy(GETUP_DEFAULTS)

    def __post_init__(self):
        if self.use_rough_terrain:
            # The fall bank is captured on flat ground: on a heightfield its states would start inside the terrain.
            raise ValueError("the get-up task trains on flat terrain only (train.use_rough_terrain: false)")
        super().__post_init__()
        # (env_config._apply_getup sets episode_length_s from the config's getup.episode_s, after construction)
        self.episode_length_s = float(self.getup.get("episode_s", self.episode_length_s))
