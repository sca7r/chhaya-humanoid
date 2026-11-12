# GPU runbook: training Chhaya's walking policy

Goal: spend as few paid GPU hours as possible. Do everything you can on your own VM first, keep
the instance busy only while it trains, and copy the results off before deleting it.

## 0. Before renting (on your VM)

1. Push this repository to your GitHub (see `HANDOFF.md`).
2. Run the tests once (no GPU needed), as in `HANDOFF.md`. The 11 arm tests
   (`tests/test_arm_motion.py`, `tests/test_chhaya_mujoco.py`) must pass.
3. Bake the gait table if you haven't: `(cd robot_motion_reference && python -m gait_studio.bake)`.
4. Check the battery and computer sizes and masses in `hardware/cad/parametric/chhaya_cad.py`
   match the parts you will use (`ELECTRONICS_LAYOUT.md`), and regenerate if not.

## Where to train

| Option | Works? | Cost for one run (≈ 13–20 h incl. setup) |
|---|---|---|
| RTX 3060 laptop (6 GB) | **No** for Isaac Lab: Isaac Sim needs ≥ 8 GB VRAM (NVIDIA's page says 16 GB) and crashes at startup on 6 GB cards. Use the laptop for everything else: tests, MuJoCo eval and policy gate, CAD, TensorBoard | free |
| RTX 4090 24 GB, rented (e.g. RunPod Community ≈ $0.34/h) | Yes: has RT cores and enough VRAM. Somewhat slower than the RTX PRO 6000, so expect the run to take ~1.3–1.5× longer | ≈ ₹400–700 at $0.34/h (≈ ₹1,000–1,800 on the $0.89/h Secure tier) |
| RTX PRO 6000 (₹208/h, the original plan) | Yes, fastest | ≈ ₹2,100–3,300 |
| A100 / H100 | **No**: no RT cores, Isaac Sim does not run | — |
| MuJoCo training on the laptop CPU | The repo's MuJoCo trainer is legacy and lacks the Isaac env's rewards and randomisation; not usable without a port | — |

Recommendation: rent an RTX 4090 for the smoke test (cheap), and keep it for the full run if the
smoke test's steps per second are acceptable. Prices change; check the provider's console.

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
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/chhaya.yaml --headless --num_envs 256
```

The startup log should show:
- `[biped_env] policy joints (12): [...]`, then the arm joints (8), and no error about joints not
  being in the articulation;
- action size **20** and actor observation size **370** (5 × 74);
- two actuator groups, `legs` and `arms`;
- after a few hundred iterations, rewards that are finite and changing. Check TensorBoard
  (`tensorboard --logdir runs --bind_all`, then open port 6006 through the provider's tunnel or
  `ssh -L 6006:localhost:6006`).

If it crashes, fix it on the instance before training. Typical suspects:
- a joint or body name regex in `envs/assets/robotV2/chhaya_robot.py`;
- the `randomize_arm_mass` event in `src/isaaclab/env_config.py`;
- a tensor shape in `envs/isaaclab/biped_env.py` (`_pjsel` / `_arm_list`).

Also watch one run with the viewer, or with `--video`, for a minute: early in training the arms
should twitch with the policy's exploration, not hang rigid.

## 4. Full run

```bash
nohup python src/isaaclab/train.py --task BipedRobotV2 --config_path config/chhaya.yaml --headless \
      > train.log 2>&1 &
```

Use `nohup` or `tmux` so a dropped SSH session doesn't kill the run. Checkpoints go to
`checkpoints/Chhaya_SAC*/` every `save_steps`.

**When to stop.** The author's legs-only runs took about 40–370 M steps (2.5–5.5 h at 4096 envs).
The arms add ~20–40 % simulation cost, and a 20-joint policy starting from scratch needs more
steps than a 12-joint one: plan for roughly 1.5–2× the legs-only numbers. Stop when, for about
30–60 min:
- `track_lin_vel_xy_exp` and the episode length have plateaued, and
- terminations (`Episode_Termination/*`) are no longer falling.

Don't run to the config's `steps` limit just because it's there.

**Expected cost:**
- setup ≈ 1 h, smoke test and fixes ≈ 0.5–1 h, run ≈ 8–14 h;
- that's ≈ 10–16 h, so about ₹2,100–3,300 at ₹208/h. Set a budget alarm at ₹4,500.
- These are estimates from the legs-only runs, not measurements: nobody has trained this robot
  yet. Expect a second run after tuning the arm reward weights.

## 5. Before deleting the instance

```bash
tar czf results.tgz checkpoints/Chhaya_SAC* runs/ train.log config/chhaya.yaml
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
