# Handoff: Chhaya (state on 2026-11-11)

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

## Setting up on your VM

```bash
git clone --recurse-submodules https://github.com/<you>/chhaya.git && cd chhaya/training
# Python 3.11 venv with the repo's pins. numpy MUST be 1.26.x: upstream's MuJoCo env does
# int(model.joint(n).qposadr), which fails on numpy 2 / MuJoCo 4.
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest tqdm       # torch: use the CPU wheel on a VM without a GPU
(cd robot_motion_reference && python -m gait_studio.bake)            # bakes the gait table the env loads
pytest tests --ignore=tests/test_env.py --ignore=tests/test_transformer.py
python -m chhaya.physics_check --out /tmp/phys                   # all checks pass
```

Expected test result: 335 pass, 7 fail. The 7 are upstream's own (gait-table and config
expectations; they fail identically on the untouched upstream import), and the two ignored files
import modules that upstream doesn't ship.

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

## Next steps

0. Before printing: set the battery and computer sizes/masses in `chhaya_cad.py` to your parts and run `hardware/regenerate.sh`.
1. GPU: follow `GPU_RUNBOOK.md` (smoke test first: the Isaac code has never run).
2. If the smoke test fails: fix it on the instance, then commit and push.
3. Adapt `export_onnx.py` and `src/mujoco/policy_gate.py` to the 74-dim frame / 20 actions, then
   export, verify and gate on the VM.
4. On-robot runtime: BipedRobotJetson reads and commands only the 12 leg servos; it needs the 8 arm
   servos added (same bus, same protocol) and the claws.
5. Later: adapt `check_actuator.py` / `jitter_probe.py` / `make_fall_bank.py` and the get-up task
   to the arms; tasks for reaching and grasping (none exist).

## Things to verify on real hardware before printing

- Horn screw circle `horn_pcd` = 14 mm (assumed): measure your horns.
- The claw servo pocket assumes an MG90S body of 22.8 x 12.2 mm: measure yours (`GR_L`, `GR_W`).
- The M2 face-hole pattern was measured from the reference STEP. Check it against one real servo
  with a test print of `l_elbow_link`, which is small.
