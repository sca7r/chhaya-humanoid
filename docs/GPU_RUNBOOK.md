# GPU runbook: training Chhaya's walking policy

Goal: spend as few paid GPU hours as possible. Do everything you can on your own VM first, keep
the instance busy only while it trains, and copy the results off before deleting it.

## 0. Before renting (on your VM)

1. Clone this repository with its submodules on the GPU machine.
2. Run the tests once (no GPU needed): `pytest tests` from `training/`.
3. Bake the gait table if you haven't (`chhaya.yaml`, `chhaya_lite.yaml` and some unit tests read it). Always pass the style: a
   plain `python -m gait_studio.bake` uses a different style and gives a different table: `(cd robot_motion_reference && python -m gait_studio.bake --style processed/gait_table_style.yaml --out processed/gait_table.npz)`.
4. Check the battery and computer sizes and masses in `hardware/cad/parametric/chhaya_cad.py`
   match the parts you will use, and regenerate if not (`hardware/regenerate.sh`).

## Where to train

Since 2026-11-14 the training runs on Isaac Lab 3.0 with the Newton backend (`ISAACLAB3.md`). It no
longer needs Isaac Sim, RT cores or a large disk, so the choice of GPU is much wider than before.

| Option | Works? | Notes |
|---|---|---|
| RTX 3060 laptop (6 GB) | Yes by NVIDIA's requirements; **not yet measured** | free. Start with 1024 robots and a 300 k replay buffer. Mains power, cooling stand, overnight runs |
| RTX 4090 24 GB, rented (≈ ₹32/h on RunPod Community) | Yes | the reference choice for a full-size run (4096 robots) |
| L4, A30, A100, H100, H200 (JarvisLabs and others) | Yes: RT cores are no longer needed | pick by price per hour; the simulation is the bottleneck, so a big data-centre card is not automatically faster per rupee |
| CPU only | Yes, for tests | ≈ 30 robot-steps/s on 2 cores: checks the code, cannot train |

Nobody has run this on a GPU yet, so steps per second and the run time are unknown. Measure them in the
smoke test below before committing to a long run. Prices change; check the provider's console.

**Storage:** about 30 GB (Isaac Lab 3.0 environment ≈ 8 GB, caches ≈ 5 GB, repo, checkpoints and logs
under 2 GB). On a rented machine put everything on the persistent volume.

## 1. Machine

| Choice | Pick |
|---|---|
| GPU | any NVIDIA card of the RTX 20 generation or newer, driver 580.65+ |
| Disk | 30 GB |
| OS | Ubuntu 22.04 or 24.04 (or Windows with WSL2) |

## 2. Install

```bash
git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && uv sync && source .venv/bin/activate      # Python 3.12, Newton backend, no Isaac Sim
git clone --recurse-submodules https://github.com/sca7r/chhaya-humanoid.git && cd chhaya-humanoid/training
uv pip install -r requirements-isaaclab3.txt
(cd robot_motion_reference && python -m gait_studio.bake --style processed/gait_table_style.yaml --out processed/gait_table.npz)
```

## 3. Smoke test: about 10 minutes, do not skip

The environment has run on a CPU only. Check it on the GPU at small scale first:

```bash
python src/isaaclab/env_check.py --num_envs 1024 --steps 200     # prints robot-steps/s
python src/isaaclab/train.py --task Chhaya --config_path config/chhaya.yaml --num_envs 1024
```

`env_check.py` should print 20 joints, 21 bodies, 2.423 kg, action size **20**, observation **370**
(5 × 74), foot forces that sum to about 25 N when standing, and no falls at zero action. In training,
rewards should be finite and changing after a few hundred iterations (`tensorboard --logdir runs`).

Then raise `--num_envs` (2048, 4096) until GPU memory runs out, and note the steps per second: that
number decides whether the laptop is enough or a rented card is worth it. If the log reports
`nconmax` / `njmax` overflow, raise them in `BipedEnvCfg.sim` (`envs/isaaclab/biped_env_cfg.py`).

If it crashes, fix it on that machine before training.

## 4. Full run

```bash
nohup python src/isaaclab/train.py --task Chhaya --config_path config/chhaya.yaml \
      > train.log 2>&1 &
```

Use `nohup` or `tmux` so a dropped SSH session doesn't kill the run. Checkpoints go to
`checkpoints/Chhaya_SAC*/` every `save_steps`.

**When to stop.** Stop when, for about 30–60 min:
- `track_lin_vel_xy_exp` and the episode length have plateaued, and
- terminations (`Episode_Termination/*`) are no longer falling.

Don't run to the config's `steps` limit just because it's there. Expect a second run after tuning: the
reward weights were tuned on PhysX and the arm weights have never been tuned.

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
