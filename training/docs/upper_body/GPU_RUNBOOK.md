# GPU runbook: training Chhaya's walking policy

Goal: spend as few paid GPU hours as possible. Do everything you can on your own VM first, keep
the instance busy only while it trains, and copy the results off before deleting it.

## 0. Before renting (on your VM)

1. Push this repository to your GitHub (see `HANDOFF.md`).
2. Run the tests once (no GPU needed), as in `HANDOFF.md`. The 11 upper-body tests
   (`tests/test_arm_motion.py`, `tests/test_upper_body_mujoco.py`) must pass.
3. Bake the gait table if you haven't: `(cd robot_motion_reference && python -m gait_studio.bake)`.
4. Optional, saves hours: if you have a legs-only checkpoint from `config.yaml` with the same
   `action_map`/home, keep it ready to upload. You can resume it on the robot with arms.

## 1. Instance

| Choice | Pick | Why |
|---|---|---|
| GPU | **RTX PRO 6000** (₹208/h) | Isaac Sim needs RT cores, which the H100/H200 don't have. It is also the cheapest of the three |
| Disk | **100 GB** | Isaac Sim + Isaac Lab ≈ 30–40 GB, checkpoints and logs a few GB |
| OS | Ubuntu 22.04 or 24.04 with a recent NVIDIA driver | the Isaac Sim pip install wants a recent glibc and driver |
| Image | one with Isaac Sim / Isaac Lab preinstalled, if the provider offers it | skips ~30–45 min of paid setup |

## 2. Install (only if the image doesn't have Isaac Lab)

The code uses the `isaaclab.*` imports, so it needs **Isaac Lab 2.x** (2.3.x with Isaac Sim 5.x,
Python 3.11, is the expected match). Don't use the 3.0 betas. Follow the official
"Installation using Isaac Sim Pip Package" page for that version. In short:
1. Make a Python 3.11 venv.
2. `pip install` the `isaacsim` package from NVIDIA's index.
3. Clone IsaacLab at the 2.3 tag and run `./isaaclab.sh --install`.

Then the repo:
```bash
git clone --recurse-submodules https://github.com/<you>/chhaya.git && cd chhaya/training
(cd robot_motion_reference && python -m gait_studio.bake)
grep -v -E '^(torch|torchvision|torchaudio)' requirements.txt > /tmp/req.txt   # keep Isaac Lab's own torch
pip install -r /tmp/req.txt
```

## 3. Smoke test: about 10 minutes, do not skip

The Isaac side of the arm code has never run (this was built without a GPU). Check it at small
scale before paying for hours:

```bash
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/config_upper_body.yaml --headless --num_envs 256
```

The startup log should show:
- `[biped_env] policy joints (12): [...]`, then the arm joints (8), and no error about joints not
  being in the articulation;
- the action and observation sizes are the **same** as a `config.yaml` run;
- two actuator groups, `legs` and `arms`;
- after a few hundred iterations, rewards that are finite and changing. Check TensorBoard
  (`tensorboard --logdir runs --bind_all`, then open port 6006 through the provider's tunnel or
  `ssh -L 6006:localhost:6006`).

If it crashes, fix it on the instance before training. Typical suspects:
- a joint or body name regex in `envs/assets/robotV2/biped_robot_upper.py`;
- the `randomize_arm_mass` event in `src/isaaclab/env_config.py`;
- a tensor shape in `envs/isaaclab/biped_env.py` (`_pjsel` / `_arm_list`).

Also watch one run with the viewer, or with `--video`, for a minute: the arms should move (swing,
random, hold) and not hang limp.

## 4. Full run

```bash
nohup python src/isaaclab/train.py --task BipedRobotV2 --config_path config/config_upper_body.yaml --headless \
      [--checkpoint <legs_only_model.pt>] > train.log 2>&1 &
```

Use `nohup` or `tmux` so a dropped SSH session doesn't kill the run. Checkpoints go to
`checkpoints/Chhaya_SAC*/` every `save_steps`.

**When to stop.** The author's legs-only runs took about 40–370 M steps (2.5–5.5 h at 4096 envs).
The arms add ~20–40 % simulation cost. Stop when, for about 30–60 min:
- `track_lin_vel_xy_exp` and the episode length have plateaued, and
- terminations (`Episode_Termination/*`) are no longer falling.

Don't run to the config's `steps` limit just because it's there.

**Expected cost:**
- setup ≈ 1 h, smoke test ≈ 0.25 h, run ≈ 5–8 h;
- that's ≈ 7–9 h, so about ₹1,500–1,900. Set a budget alarm at ₹2,500.
- Resuming a legs-only checkpoint usually cuts the run substantially.

## 5. Before deleting the instance

```bash
tar czf results.tgz checkpoints/Chhaya_SAC* runs/ train.log config/config_upper_body.yaml
```

Download `results.tgz` (scp, or the provider's file browser) and **check it opens on your side**.
Only then delete the instance. Storage keeps billing until it's deleted.

## 6. After (on your VM or laptop, no GPU)

```bash
python -m src.isaaclab.export_onnx --checkpoint checkpoints/<run>/step_<N>.pt
python -m src.isaaclab.verify_onnx_bundle --bundle exported/<bundle>
python -m src.mujoco.policy_gate --bundle exported/<bundle>           # arms at rest
```

The gate builds the MuJoCo env from the run's own `config.yaml` (the copy next to the checkpoint),
so the arms are there automatically. To check walking while the arms move, set
`mujoco_arm_mode: swing` (then `random`) in that copy and gate the raw checkpoint:
`python -m src.mujoco.policy_gate --checkpoint checkpoints/<run>/step_<N>.pt` (see the gate's
`--help` for the exact flag). `python src/mujoco/sim2real.py --checkpoint ... --vx 0.1`
shows it in the MuJoCo viewer.
