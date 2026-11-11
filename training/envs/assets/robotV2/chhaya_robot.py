"""
Isaac Lab asset for Chhaya: the legs plus 8 arm joints (claws and halo head are fixed mass).

chhaya.usd is a thin layer over robot.usd (built by hardware/tools/build_usd.py
with plain usd-core): the legs are the exact prims the current policies were trained on, the chest and
head are merged into base_link (as the URDF importer merged imu_link) and the arms follow the importer's
conventions. Prim layout is unchanged: /World/Robot/<link>, so the scene's contact sensor
("{ENV_REGEX_NS}/Robot/Robot/.*") and IMU ("{ENV_REGEX_NS}/Robot/Robot/base_link") paths still work.

Two BAM actuator groups with the same servo model: "legs" (first, so code that reads
bam_actuators[0] keeps reading the legs) and "arms".
"""
import os

from envs.assets.robotV2.biped_robot import BIPED_ROBOT_CFG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH_CHHAYA = os.path.join(CURRENT_DIR, "chhaya.usd")

LEG_JOINT_REGEX = r"(l|r)_(hip_yaw|hip_roll|hip_pitch|knee|ankle_roll|ankle_pitch).*"
ARM_JOINT_REGEX = r"(l|r)_(shoulder_pitch|shoulder_roll|arm_yaw|elbow)"
# L,R interleaved per type, like the legs (and like chhaya.model.ARM_JOINTS on the MuJoCo side)
ARM_JOINT_NAMES = ["l_shoulder_pitch", "r_shoulder_pitch", "l_shoulder_roll", "r_shoulder_roll",
                   "l_arm_yaw", "r_arm_yaw", "l_elbow", "r_elbow"]
ARM_BODY_REGEX = r"(l|r)_(shoulder|upper_arm|elbow|forearm)_link"
# arm rest pose = home/reset pose of the arms (chhaya/motions.py REST)
ARM_REST = {"l_shoulder_pitch": 0.0, "r_shoulder_pitch": 0.0, "l_shoulder_roll": 0.12, "r_shoulder_roll": 0.12,
            "l_arm_yaw": 0.0, "r_arm_yaw": 0.0, "l_elbow": -0.35, "r_elbow": -0.35}

_legs = BIPED_ROBOT_CFG.actuators["legs"]

CHHAYA_ROBOT_CFG = BIPED_ROBOT_CFG.replace(
    spawn=BIPED_ROBOT_CFG.spawn.replace(usd_path=USD_PATH_CHHAYA),
    init_state=BIPED_ROBOT_CFG.init_state.replace(
        joint_pos={**BIPED_ROBOT_CFG.init_state.joint_pos, **ARM_REST}),
    actuators={
        "legs": _legs.replace(joint_names_expr=[LEG_JOINT_REGEX]),
        "arms": _legs.replace(joint_names_expr=[ARM_JOINT_REGEX]),
    },
)
