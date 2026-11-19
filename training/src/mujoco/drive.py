"""drive.py -- drive an exported policy in the MuJoCo viewer. Tap a spot and the robot walks there.

    python -m src.mujoco.drive --bundle exported/chhaya_final

  Right double-click the floor   walk to that spot (a red marker shows it)
  I / M                          walk forward / backward
  J / L                          turn left / right
  U / O                          step left / right
  K                              stop

Every input becomes the one command the policy knows (vx, vy, wz), limited to the training range
(utils/teleop.py). Voice or a phone UI only need to produce the same command.
"""
import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.mujoco.policy_gate import Policy, _run_config_path
from envs.mujoco.biped_env_v2 import STEP_DT, BipedEnvV2
from utils import Config
from utils.teleop import axes_to_command, go_to_axes

KEYS = {ord("I"): (1, 0, 0), ord("M"): (-1, 0, 0), ord("J"): (0, 0, 1), ord("L"): (0, 0, -1),
        ord("U"): (0, 1, 0), ord("O"): (0, -1, 0), ord("K"): (0, 0, 0)}
LEVEL = 0.75        # fraction of the training command range used by the keys and the go-to


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="exported policy bundle (train_supermimic.py --export)")
    a = p.parse_args()
    policy = Policy(a.bundle)
    env = BipedEnvV2(history_size=policy.hist, use_history=True, critic_has_privileged_info=True,
                     env_config=Config(_run_config_path(a.bundle)).train.env_config)
    ranges, shape = env.command_limit()
    obs, _ = env.reset()
    state = {"axes": (0, 0, 0), "target": None}

    def key(k):
        if k in KEYS:
            state["axes"], state["target"] = KEYS[k], None

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key) as v:
        look = v.cam.lookat.copy()
        while v.is_running():
            t0 = time.time()
            # ponytail: the viewer's right double-click centres the camera on the clicked point, so a jump of
            # cam.lookat onto the floor (z = 0) is the tap. A pan that ends exactly at floor height reads as one too.
            if np.linalg.norm(v.cam.lookat - look) > 0.01 and abs(v.cam.lookat[2]) < 1e-3:
                state["target"] = v.cam.lookat[:2].copy()
            look = v.cam.lookat.copy()
            if state["target"] is not None:
                w, x, y, z = env.data.qpos[3:7]
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                state["axes"] = go_to_axes(*env.data.qpos[:2], yaw, *state["target"])
            env.set_command(*axes_to_command(*state["axes"], ranges, shape, LEVEL))
            obs, _, term, _, _ = env.step(policy(obs["actor"]))
            if term:
                obs, _ = env.reset()
            with v.lock():
                v.user_scn.ngeom = 0
                if state["target"] is not None:
                    mujoco.mjv_initGeom(v.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0],
                                        [*state["target"], 0.002], np.eye(3).ravel(), [0.9, 0.2, 0.2, 1])
                    v.user_scn.ngeom = 1
            v.sync()
            time.sleep(max(0.0, STEP_DT - (time.time() - t0)))


if __name__ == "__main__":
    main()
