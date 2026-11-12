# Where and how to train Chhaya (checked 2026-11-12)

## What the current training needs

| Need | What the code uses today |
|---|---|
| Simulator | Isaac Lab 2.x on Isaac Sim (PhysX), `DirectRLEnv` (`training/envs/isaaclab/biped_env.py`, ~2,500 lines) |
| GPU | Isaac Sim: RT cores and ≥ 16 GB VRAM by NVIDIA's spec (8 GB is the practical floor; 6 GB crashes at start) |
| Parallel robots | 4096, physics 200 Hz, policy 50 Hz, 20 s episodes |
| Robot | 20 joints, `chhaya.usd` (Isaac) and `chhaya.xml` (MuJoCo, same geometry and masses) |
| Servo model | BAM STS3215 model, pure PyTorch (`training/actuator/bam_kernel.py`): portable to any simulator |
| Terrain | rough terrain generator (randomisation, no curriculum) |
| Sensors | foot contact forces and air time, IMU (step-averaged accelerometer, gyro), joint encoders (quantised) |
| Randomisation | masses, friction, COM, payload, action delay, servo parameters and voltage, sensor noise and bias, pushes |
| Reference motion | baked gait table (`robot_motion_reference`, `gait_studio.bake`) for the imitation rewards |
| RL | own PyTorch SAC: 1 M replay, batch 2048, 5-step returns, mirror-equivariant actor, symmetry augmentation, CAPS smoothness, 5-frame history |
| Memory | replay buffer ≈ 5 GB at 1 M transitions (370-dim history obs, stored twice + critic obs) |
| After training | MuJoCo eval and policy gate (CPU), ONNX export, on-robot runtime |

## Options

| Option | Runs on the RTX 3060 laptop (6 GB)? | Porting work | Notes |
|---|---|---|---|
| **A. Current code on a rented RTX 4090** | no (rented) | none | ≈ ₹400–700 per run. Fastest path to a first policy |
| **B. Isaac Lab 3.0, kit-less Newton (MuJoCo Warp solver)** | not ruled out by NVIDIA's docs; no published VRAM figure for kit-less | medium: 2.x → 3.0 breaking changes (quaternion order x,y,z,w, data proxies, launcher, importers) and Newton feature gaps | Keeps the Isaac Lab structure. Newton is still "under active development" per NVIDIA |
| **C. mjlab 1.6 (MuJoCo Warp, Isaac Lab-style API, Apache-2.0)** | yes in principle: needs any NVIDIA CUDA GPU, no RT cores, no Isaac Sim | large: rewrite the env as mjlab terms; reuse reward functions, BAM kernel and SAC | Has terrains, rich randomisation, custom GPU actuators, actuation delay, contact sensors. Training sim = eval sim (MuJoCo), which removes the PhysX→MuJoCo gap |
| **D. Holosoma (Amazon FAR) + FastSAC on MuJoCo Warp** | likely, with fewer envs | large: humanoid framework built around G1/T1 | FastSAC trains humanoid walking in ~15 min on one RTX 4090. Off-policy like our SAC |

**Tested here (no GPU, Warp on CPU):** `chhaya.xml` loads in MuJoCo 4.0.1 and MuJoCo Warp 4.0.1 and steps
16 worlds; the result matches CPU MuJoCo to 7 digits. Colliders: 37 meshes, 12 boxes, 1 plane. The torso
mesh has 15,000 vertices: replace it with boxes before GPU training (mesh collisions are the slow path in
MuJoCo Warp).

## Making it fit 6 GB and run faster

- 1024–2048 parallel robots instead of 4096.
- Replay buffer 300 k transitions (≈ 1.6 GB), or keep it in CPU RAM, or store it in fp16.
- FastSAC recipe (from the 15-minute paper): batch up to 8 K, more gradient steps per env step, layer
  norm, distributional (C51) critic, γ 0.97 for velocity tracking, Adam β₂ 0.95, weight decay 1e-3.
- Simple collision shapes (boxes and capsules) for the torso, chest and feet.
- Expect a laptop RTX 3060 to be several times slower than a desktop 4090 (about 1/3 of its memory
  bandwidth and 1/5 of its compute) and to throttle when hot: plan overnight runs.

## Laptop setup for options B–D

Ubuntu 22.04/24.04 (or Windows with WSL2), NVIDIA driver recent enough for CUDA 12.8+, Python 3.10–3.13,
16 GB RAM or more, ~20 GB disk, the laptop on mains power and on a cooling stand.

## Recommendation

1. Now: option A. The code is ready; one rented 4090 run gives the first policy for little money.
2. In parallel: port to option C (mjlab), keeping our SAC, reward functions and BAM kernel. Once it
   reproduces option A's policy quality in MuJoCo eval, later iterations run on the laptop for free.
3. Adopt the FastSAC changes in whichever path is used; they cut training time the most.

Sources: [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html),
[Isaac Lab kit-less installation](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/kitless_installation.html),
[Isaac Lab 3.0 release notes](https://docs.nvidia.com/isaaclab/3.0/references/release-notes),
[Isaac Lab 3.0 overview](https://physicalailab.io/en/isaac-lab-3-0-release-whats-new-requirements-migration/),
[mjlab on GitHub](https://github.com/mujocolab/mjlab), [mjlab on PyPI](https://pypi.org/project/mjlab/),
[mjlab paper](https://arxiv.org/pdf/2601.22074v2),
[Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](https://arxiv.org/html/2512.01996v1),
[Holosoma announcement](https://www.humanoidsdaily.com/news/amazon-far-open-sources-holosoma-to-unify-humanoid-simulation-and-training).
