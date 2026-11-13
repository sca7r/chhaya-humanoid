# Isaac Lab 3.0 for Chhaya (checked 2026-11-13)

Checked against Isaac Lab **3.0.0** (tag `v3.0.0`; Newton 1.6.1, Warp 1.17, PyTorch 2.11, Python 3.12),
by installing it without Isaac Sim and running Chhaya in it on a CPU. No GPU was available, so speed and
memory on a GPU are not measured.

## What 3.0 changes for us

- **Isaac Sim is optional.** The default physics is Newton with the MuJoCo-Warp solver. Headless Newton
  needs a CUDA GPU of the RTX 20 generation or newer (driver 580.65+), **no RT cores** and none of Isaac
  Sim's 16 GB VRAM requirement. NVIDIA states no VRAM minimum for this mode.
- So the RTX 3060 laptop qualifies, and so do the A30, A100, H200 and L4.
- The stock rough-terrain walking tasks (G1, H1, Go2, Cassie, ANYmal-D) default to Newton, with generated
  terrain, contact sensors, ray casters and mass/friction/COM randomisation: the features we use.
- Training would use the same solver family as our MuJoCo evaluation, instead of PhysX.

## Test result

`training/experimental/isaaclab3/` (`make_newton_usd.py`, `smoke_test.py`): Chhaya as an articulation in
a minimal `DirectRLEnv` on generated rough terrain, Newton + MuJoCo-Warp, 2 robots, CPU.

| Check | Result |
|---|---|
| Loads | 20 joints, 21 bodies, 2.567 kg (matches the model) |
| Stands for 4 s under PD control | yes, no falls, base height 0.284 m (CPU MuJoCo: 0.279 m) |
| Contact sensor | vertical force 25.2 N per robot = the robot's weight |
| Speed | 51 robot-steps/s on 2 CPU cores (says nothing about a GPU) |

## What had to change in the robot file

`chhaya.usd` as built for PhysX does **not** work on Newton. `make_newton_usd.py` fixes three things:

| Problem | Effect on Newton | Fix |
|---|---|---|
| The leg colliders have their collision flag on a parent of the mesh | Newton skips them: the legs have no collision and the robot falls through the ground | move the flag onto the meshes |
| Leg colliders ask for convex decomposition | ~950 collision pieces per robot | convex hulls (45 pieces) |
| Materials reference Isaac Sim's built-in OmniPBR | cannot be resolved without Isaac Sim | drop the reference |

Still to do on the asset: neighbouring leg hulls overlap (calf/foot, hip/thigh, torso/hip), which
creates constant self-contacts. They need filter pairs or simple box colliders.

## What has to change in the training code

Counted in `training/envs/isaaclab`, `envs/assets/robotV2`, `src/isaaclab`:

| Change in 3.0 | Our code |
|---|---|
| Data fields return `ProxyArray`: read with `.torch` | 106 reads |
| Quaternions are x,y,z,w instead of w,x,y,z | ~176 places touch quaternions; errors here are silent |
| `write_*_to_sim` → `write_*_to_sim_index(...)`, joint targets → `actuators.target_command` | 12 calls |
| `root_physx_view` is gone (masses, COMs, mass matrix) | 6 uses, incl. the BAM servo model's mass matrix (`data.mass_matrix` exists on Newton) |
| Actuator classes: limit fields renamed, group `armature`/`friction` properties removed | `BamActuator` needs a rewrite of its setup |
| `Imu` now gives only angular velocity and linear acceleration (the old one is `Pva`) | fits what we read; 16 lines to check |
| Contact sensor: `net_forces_w` now includes friction | 66 uses to review |
| Physics config moves to `PhysxCfg` / `NewtonCfg` presets; `AppLauncher`, `--headless` replaced | 48 lines |
| Joint order under Newton is per-limb (depth-first), not PhysX's breadth-first | our code finds joints by name: fine |
| MuJoCo-Warp does not enforce joint speed limits | our servo model limits speed itself: fine |
| Python 3.12, NumPy ≥ 2, PyTorch 2.11 | our MuJoCo env needs a small NumPy 2 fix |
| A single flat generated tile fails ("mesh volume is too small") | our terrain mixes flat with rough tiles; to verify |

Estimate: about a week of work. The rewards were tuned on PhysX, so expect some retuning on Newton.

## Verdict

Isaac Lab 3.0 with Newton is the best route to training on the laptop, and a smaller port than mjlab
because the environment structure (`DirectRLEnv`, sensors, terrain, events) stays. A further advantage:
it runs on a CPU, so the ported environment can be tested end to end without renting a GPU, which the
current Isaac Sim code cannot.

Unknown until tried on a GPU: steps per second and how many robots fit in 6 GB.
