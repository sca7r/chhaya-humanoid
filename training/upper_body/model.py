"""Paths and joint tables for the upper-body robot models."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # the BipedRobot checkout

LEG_JOINTS = [
    "l_hip_yaw", "r_hip_yaw",
    "l_hip_roll_joint", "r_hip_roll_joint",
    "l_hip_pitch_joint", "r_hip_pitch_joint",
    "l_knee_joint", "r_knee_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint",
]

# L,R interleaved like the legs
ARM_JOINTS = [
    "l_shoulder_pitch", "r_shoulder_pitch",
    "l_shoulder_roll", "r_shoulder_roll",
    "l_arm_yaw", "r_arm_yaw",
    "l_elbow", "r_elbow",
]

# Same limits as the URDF/MJCF (rad). Both arms share sign conventions:
# +roll = arm out, -pitch = arm forward/up, -elbow = flex, +yaw = upper arm rotates outward.
ARM_LIMITS = {
    "shoulder_pitch": (-3.0, 1.0),
    "shoulder_roll": (0.05, 2.6),
    "arm_yaw": (-1.57, 1.57),
    "elbow": (-2.0, 0.1),
}


def arm_limits(joint: str):
    return ARM_LIMITS[joint[2:]]


def model_path(variant: str) -> Path:
    return REPO / "envs/assets/robotV2" / f"robot_v2_upper_{variant}.xml"
