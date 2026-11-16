"""Chhaya Lite, no policy: does the robot stand still with zero action, as a function of the two
unknowns that matter most, gear play and controller stiffness? MuJoCo, 10 s per case.

    python -m chhaya.lite_stand_scan            (from training/)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from utils import Config  # noqa: E402
from envs.mujoco.biped_env_v2 import BipedEnvV2  # noqa: E402


def stand(cfg_name, lash=None, kp=None, seconds=10.0, seed=0):
    ec = Config(os.path.join(ROOT, "config", cfg_name)).train.env_config
    if lash is not None:
        ec.backlash["mujoco_lash_deg"] = lash
        ec.backlash["enabled"] = lash > 0
    if kp is not None:
        ec.actuator["kp_fw"] = kp
    env = BipedEnvV2(history_size=5, use_history=True, env_config=ec, seed=seed)
    env.reset(seed=seed)
    env.set_command(0.0, 0.0, 0.0)
    n = env.action_space.shape[0]
    qv, fell = [], None
    for k in range(int(seconds * 50)):
        _, _, term, _, _ = env.step(np.zeros(n, np.float32))
        if term:
            fell = k / 50.0
            break
        qv.append(env.data.qvel[6:18].copy())
    rms = float(np.sqrt((np.array(qv[25:]) ** 2).mean())) if len(qv) > 30 else float("nan")
    return fell, rms


def main():
    f, r = stand("chhaya.yaml")
    print(f"Chhaya (STS3215, as configured): fell {f}, leg joint jitter {r:.3f} rad/s")
    print("\nChhaya Lite. Cell = time of fall [s] ('-' = stood 10 s) / leg joint jitter [rad/s]")
    kps = (70, 110, 170)
    print("| gear play | " + " | ".join(f"full drive at {0.97 / (kp * 0.166) * 57.3:.1f} deg" for kp in kps) + " |")
    print("|---|" + "---|" * len(kps))
    for lash in (0.0, 0.5, 1.0, 2.0, 3.0):
        cells = []
        for kp in kps:
            f, r = stand("chhaya_lite.yaml", lash, kp)
            cells.append(f"{'-' if f is None else f'{f:.1f}'} / {r:.3f}")
        print(f"| {lash:.1f} deg | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
