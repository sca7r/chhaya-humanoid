# Training on Isaac Lab 3.0 (Newton, no Isaac Sim)

Since 2026-11-14 the training code runs on **Isaac Lab 3.0.0** with the **Newton** backend and the
**MuJoCo-Warp** solver. Isaac Sim (Omniverse, PhysX) is no longer used.

| | Before (Isaac Lab 2.x) | Now (Isaac Lab 3.0) |
|---|---|---|
| Simulator | Isaac Sim, PhysX | Newton, MuJoCo-Warp solver |
| GPU | RT cores, ≥ 8–16 GB VRAM | any CUDA GPU of the RTX 20 generation or newer (driver 580.65+); no RT cores, no stated VRAM minimum |
| RTX 3060 laptop (6 GB) | does not start | supported by NVIDIA's requirements; **not yet measured** |
| A100 / H100 / H200 / A30 | do not run Isaac Sim | usable |
| CPU | no | yes, slowly: the whole pipeline is tested on a CPU |
| Install size | ≈ 30–40 GB | ≈ 8 GB |
| Parallel robots, one shared policy | yes | yes (`--num_envs`) |

## Status

Tested here on a 2-core CPU (no GPU was available):

| Check (`config/chhaya.yaml`, whole-body policy, 20 joints) | Result |
|---|---|
| Real environment (`BipedEnv`) builds | 20 joints, 21 bodies, 2.423 kg; obs 370 (5 × 74), critic 84, 20 actions |
| Stands at zero action, rough terrain, randomisation on | base 0.27–0.28 m, foot forces sum to 25.1 N (weight 25.2 N), peak servo torque 0.5 N·m |
| IMU | 9.81 m/s² up at rest, gravity vector (0, 0, −1) |
| BAM servo model | effective inertia 0.021–0.026 kg·m² (armature 0.023 + link inertia) |
| Self-collision | on: 186 of 210 link pairs can collide (hands–torso, foot–foot included); no spurious contacts when standing |
| 320 steps with random actions | pushes, falls, resets and all randomisation events run; observations stay finite |
| Flat terrain | works; floor part of the foot forces is reported separately |
| `train.py`, SAC, 4 robots, 220 iterations | runs, loss finite, checkpoints and config saved |
| `play.py` from that checkpoint | runs |
| Unit tests | all pass: 360 on Python 3.11, 366 on 3.12 |
| Speed | ≈ 30–40 robot-steps/s on 2 CPU cores (says nothing about a GPU) |

**Not known until it runs on a GPU:** steps per second, how many robots fit in 6 GB, and whether the
reward weights (tuned on PhysX) need retuning on Newton.

## Install (laptop or rented GPU)

Ubuntu 22.04/24.04 (or WSL2), NVIDIA driver 580.65 or newer, [uv](https://docs.astral.sh/uv/).

```bash
git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && uv sync                 # Python 3.12 env in IsaacLab/.venv, Newton backend, no Isaac Sim
source .venv/bin/activate
cd <chhaya>/training
uv pip install -r requirements-isaaclab3.txt
```

Do not install `requirements.txt` into this environment (old pins: torch cu118, NumPy 1.26).

## Run

From `training/`:

```bash
# 1. check the environment (2 robots; add --device cuda:0 --num_envs 1024 on a GPU)
python src/isaaclab/env_check.py --num_envs 2 --steps 200

# 2. train
python src/isaaclab/train.py --task Chhaya --config_path config/chhaya.yaml --num_envs 1024

# 3. watch / evaluate a checkpoint
python src/isaaclab/play.py --checkpoint checkpoints/<run>/step_<N>.pt --num_envs 4 --viz newton --command 0.15 0 0
```

`--device` defaults to `cuda:0` when a GPU is present, else `cpu`. `--viz newton` opens a viewer;
training runs without one. `--headless` no longer exists.

Settings for a 6 GB GPU (start here, raise until memory runs out):

| Setting | 6 GB laptop | 24 GB rented |
|---|---|---|
| `--num_envs` | 1024 | 4096 |
| `buffer.size` in `config/chhaya.yaml` | 300000 (≈ 1.6 GB) | 1000000 (≈ 5 GB) |
| `buffer.batch_size` | 1024 | 2048 |

If the log prints `Number of Newton contacts exceeded MJWarp limit` or `nconmax`/`njmax` overflow, raise
`njmax` / `nconmax` in `BipedEnvCfg.sim` (`envs/isaaclab/biped_env_cfg.py`). They are per robot.

## How the port works

Isaac Lab 3.0 changed two things that touch nearly every line of the environment: data fields return a
`ProxyArray` instead of a tensor, and quaternions are (x, y, z, w) instead of (w, x, y, z). MuJoCo, the
rewards, the gait tables and the on-robot runtime all use (w, x, y, z).

The environment therefore keeps tensors and (w, x, y, z) inside, and converts once at the simulator:

| File | Role |
|---|---|
| `envs/isaaclab/compat.py` | `TorchData` (`env.rd` robot data, `env.cs` contact data: tensors, quaternions converted) and the writers (`write_root_pose`, `write_joint_state`, `set_joint_targets`, …) |
| `envs/isaaclab/quat.py` | the quaternion functions the env uses, in (w, x, y, z), pure torch; checked against MuJoCo in `tests/test_isaaclab3_compat.py` |
| `src/isaaclab/launch.py` | start-up: `launch_simulation` replaces `AppLauncher` |

What else changed:

| Area | Change |
|---|---|
| Robot file | `chhaya_newton.usd` / `robot_newton.usd`, made from the PhysX files by `envs/assets/robotV2/make_newton_usd.py` (`hardware/tools/build_usd.py` runs it): collision flags moved onto the meshes, convex hulls, no Omniverse material |
| Physics config | `NewtonCfg` + `MJWarpSolverCfg`: 200 Hz step (the servo model's rate), 2 solver substeps |
| Servo model | `BamActuator`: mass matrix from `data.mass_matrix`; armature and zero joint friction set on the cfg (3.0 writes joint properties before it builds the actuator); `use_newton_actuators=False` |
| IMU | computed in `BipedEnv._read_imu` from the base state: step-averaged accelerometer at the IMU position, as before. Newton's own IMU sensor gives the last substep only and no gravity vector |
| Contact sensor | `find_sensors` / `sensor_names`; the floor filter is a shape filter (`filter_shape_prim_expr`) because the terrain is a static shape |
| Terrain | always generated. Flat ground is a slab (`FlatSlabTerrainCfg`): Newton rejects zero-thickness sheets, and the stock plane is downloaded from NVIDIA's servers. The mesh size follows the robot count (2×2 to 20×20 tiles) |
| Joint order | Newton orders joints depth-first (left leg, then right). The env maps policy joints by name (`policy_joint_names`) for both robots |
| Resets and COM randomisation | own functions in `mdp/events.py` (`reset_root_state_uniform`, `reset_joints_by_offset`, `randomize_rigid_body_com_absolute` via `set_coms_index`) |
| Removed | `physx_cache.py`, the PhysX solver settings, the training video camera, the keyboard teleop HUD in `play.py` (Omniverse UI) |
| MuJoCo env | NumPy 2 fix (`int()` on 1-element arrays), so one Python 3.12 environment runs training and evaluation |

## Not ported yet

| Item | State |
|---|---|
| `src/isaaclab/check_actuator.py`, `jitter_probe.py`, `make_fall_bank.py`, `profile_train.py`, `record_reel.py`, `teleop.py` | still use the Isaac Lab 2.x launcher; they will not start |
| On-robot runtime | not written yet for the 20-joint policy |
| Video recording during training | removed; use `play.py --viz` |

## Known differences from the PhysX version

- Contact forces now include friction (`net_forces_w`); thresholds on force norms (1 N contact flag) are
  unaffected in practice.
- MuJoCo-Warp does not enforce joint speed limits; the servo model limits speed itself.
- `randomize_payload` and `randomize_mass` both act on `base_link` and each remembers its own default
  mass at first use, so the payload term's baseline includes the first mass draw (a few grams).
- The reward weights and curriculum were tuned on PhysX. Training and evaluation now use the same solver
  family (MuJoCo), which should narrow the sim-to-sim gap, but expect some retuning.
