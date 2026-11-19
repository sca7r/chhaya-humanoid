# Handoff: Chhaya (state on 2026-11-18)

Why things are the way they are, in date order: `PROJECT_LOG.md`.

Chhaya is one robot and one project: a 20-joint humanoid with claws and a sensor head. Credits and
licences for the open-source work it builds on are in `NOTICE.md`.

## Where things are

| Path | What it is |
|---|---|
| `hardware/` | everything physical: CAD (`cad/`, with the parametric source in `cad/parametric/`), print files (`print/stl/`), parts lists (`bom/`), electronics, the robot description (`robot_description/urdf/chhaya.urdf`), the design notes (`docs/design.md`) and the build guide (`docs/build/`) |
| `hardware/tools/`, `hardware/regenerate.sh` | rebuild and check everything generated from the parametric CAD |
| `training/` | the RL code (Isaac Lab + MuJoCo). Chhaya's config is `config/chhaya.yaml`; its models are `envs/assets/robotV2/chhaya.xml` and `chhaya.usd`; `chhaya/` has the joint tables, arm motions and the physics check |
| `docs/` | `TRAINING.md` (how the policy and the arms work together), `GPU_RUNBOOK.md`, this file, `physics/` (physics check results), `BUDGET_ELECTRONICS.md` |
| `training/robot_hardware` | a link to `../hardware` (the code's mesh paths go through it) |
| `training/torch_rl_algorithms`, `training/robot_motion_reference` | submodules, pinned |

Two things keep their original names because the code and tests refer to them: the training task
`BipedRobotV2`, and the legs-only rig (`config/config.yaml`, `robot_v2.xml`, `Robot.urdf`), which is
Chhaya without the chest, head and arms. The regression tests run on the rig, and it is also how
you bring up the legs before mounting the arms.

## Setting up

Training runs on Isaac Lab 3.0 (Python 3.12): install as in `ISAACLAB3.md`. Its environment also runs
all tests. A plain Python 3.11 venv with the repo's pins runs the tests and the MuJoCo side too.

```bash
cd chhaya/training
(cd robot_motion_reference && python -m gait_studio.bake --style processed/gait_table_style.yaml --out processed/gait_table.npz)
pytest tests
python -m chhaya.physics_check --out /tmp/phys                   # all checks pass
```

The zip deliveries already contain the baked gait table, so the bake step is only needed after a
fresh `git clone` (the table is a generated file, not in git). Only the gait-reference configs and
some tests read it; `chhaya.yaml` and `chhaya_lite.yaml` do not.

Expected test result: all pass (363 on Python 3.11, 369 on 3.12, measured 2026-11-18 from the zip).
If 5–8 gait-table tests fail, the table is missing or was baked without the `--style` argument.

On Windows the `training/robot_hardware` link needs `git config core.symlinks true` (or WSL).

To regenerate the CAD and models after changing a dimension (needs `cadquery trimesh python-fcl
usd-core mujoco pillow`), run `hardware/regenerate.sh`. It runs, in order: CAD,
build_description, check_interference, strength_check, validate_model, build_usd, renders, and
writes each file to its final place (`hardware/cad/step/bodies`, `hardware/print/stl`,
`hardware/robot_description`, `training/envs/assets/robotV2`).

## Decisions made (don't revisit without a reason)

- 4 DOF per arm (pitch, roll, yaw, elbow). There is **no waist servo** (decided 2026-11-09).
- Redesign of 2026-11-11: rounded, open look (no cosmetic covers, open chest); halo head (H3: dark
  face disc in a hoop, drilled sensor plate) at the chest front; one-servo claws (MG90S class) on a
  standard wrist plate (4x M3). The claw is fixed mass in simulation, scripted on the real robot.
- Every printed part must pass `tools/strength_check.py` (safety factor >= 3 on real cross-sections)
  and `tools/check_interference.py` (>= 1.0 mm moving clearance, zero overlap) in `hardware/tools/`.
- Plug and play: every module has one standard interface (table in
  `hardware/docs/design.md`). Keep those interfaces when changing parts.
- The legs are fixed parts. A light restyle (rounding only) is allowed; joint positions, servo
  mounting and foot soles must not change.
- STS3215 on all 20 joints in simulation (the only servo with a validated BAM model).
- **Whole-body policy (decided 2026-11-12):** one policy commands all 20 joints (20 actions, 74-dim
  frame). Scripted arms (`arms.control: scripted`) remain only as a fallback. See `TRAINING.md`.
- **Electronics (2026-11-12, `ELECTRONICS_LAYOUT.md`):** chest raised 23 mm onto a printed electronics
  rack; battery and Raspberry Pi 5 under the chest, PCB and bus adapter in the torso.
  A Jetson no longer fits. Weigh the real torso before training.
- One robot, one project (decided 2026-11-11): no separate "upper body" folders, configs or docs.
- Design language: like the legs. Servos are the structure, flat plates on their M2 face holes,
  dual-support forks, filleted joints, no rods or trusses, not boxy.
- Clearances for real prints: ≥ 1.0 mm between moving parts (checked on every build). The idler
  bores are 0.3 mm running fits.
- Strength: `tools/strength_check.py` must pass (safety factor ≥ 3 at every critical section).

Two tracks: Chhaya (`config/chhaya.yaml`, STS3215) and Chhaya Lite (`config/chhaya_lite.yaml`, hobby servos,
estimated servo model): see `CHHAYA_LITE.md`.

## Next steps

0. Before printing: set the battery and computer sizes/masses in `chhaya_cad.py` to your parts and run `hardware/regenerate.sh`.
1. GPU: follow `GPU_RUNBOOK.md` (smoke test first: the Isaac Lab 3.0 code has run on a CPU only).
   Then optionally SuperMimic (`SUPERMIMIC.md`) against SAC at equal wall time.
2. If the smoke test fails: fix it on the instance, then commit and push.
3. Adapt `export_onnx.py`, `src/mujoco/policy_gate.py` and `play.py` to the 74-dim frame / 20 actions
   (and to SuperMimic checkpoints), then export, verify and gate.
4. On-robot runtime: BipedRobotJetson reads and commands only the 12 leg servos; it needs the 8 arm
   servos added (same bus, same protocol) and the claws.
5. Later: adapt `check_actuator.py` / `jitter_probe.py` / `make_fall_bank.py` and the get-up task
   to the arms; tasks for reaching and grasping (none exist).

## Things to verify on real hardware before printing

- Horn screw circle `horn_pcd` = 14 mm (assumed): measure your horns.
- The claw servo pocket assumes an MG90S body of 22.8 x 12.2 mm: measure yours (`GR_L`, `GR_W`).
- The M2 face-hole pattern was measured from the reference STEP. Check it against one real servo
  with a test print of `l_elbow_link`, which is small.
