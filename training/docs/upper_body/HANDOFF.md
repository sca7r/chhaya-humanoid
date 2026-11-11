# Handoff: Chhaya (state on 2026-11-11)

Chhaya is the Sir Servington biped (12-servo legs, by Pablo Gomez Martinez) with a new upper body:
two 4-DOF arms with one-servo claws, an open chest and a halo head.

## Where things are

One repository:

| Path | What it is |
|---|---|
| `hardware/` | Sir Servington's hardware (CAD, print files, electronics, robot description), plus `hardware/upper_body/` (CadQuery source, STEP/STL, check tools, renders, README with design, strength, printing and assembly) and the upper-body URDF and meshes in `hardware/robot_description/` |
| `training/` | Sir Servington's training code (Isaac Lab + MuJoCo), plus the upper-body integration (see `UPPER_BODY.md`), `config/config_upper_body.yaml`, the USD and MJCF, `upper_body/` (joint tables, motions, physics check), tests and these docs |
| `training/robot_hardware` | a link to `../hardware` (upstream had a submodule here; the code's mesh paths go through it) |
| `training/torch_rl_algorithms`, `training/robot_motion_reference` | upstream's submodules, pinned to the commits upstream uses |

## Setting up on your VM

```bash
git clone --recurse-submodules https://github.com/<you>/chhaya.git && cd chhaya/training
# Python 3.11 venv with the repo's pins. numpy MUST be 1.26.x: upstream's MuJoCo env does
# int(model.joint(n).qposadr), which fails on numpy 2 / MuJoCo 4.
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest tqdm       # torch: use the CPU wheel on a VM without a GPU
(cd robot_motion_reference && python -m gait_studio.bake)            # bakes the gait table the env loads
pytest tests --ignore=tests/test_env.py --ignore=tests/test_transformer.py
python -m upper_body.physics_check --out /tmp/phys                   # all checks pass
```

Expected test result: 335 pass, 7 fail. The 7 are upstream's own (gait-table and config
expectations; they fail identically on the untouched upstream import), and the two ignored files
import modules that upstream doesn't ship.

On Windows the `training/robot_hardware` link needs `git config core.symlinks true` (or WSL).

To regenerate the CAD and models after changing a dimension (needs `cadquery trimesh python-fcl
usd-core mujoco pillow`), run `hardware/upper_body/regenerate.sh`. It runs, in order: CAD,
build_description, check_interference, strength_check, validate_model, build_usd, renders, and
writes straight into `hardware/robot_description` and `training/envs/assets`.

## Decisions made (don't revisit without a reason)

- 4 DOF per arm (pitch, roll, yaw, elbow). There is **no waist servo** (decided 2026-11-09).
- Redesign of 2026-11-11: rounded, open look (no cosmetic covers, open chest); halo head (H3: dark
  face disc in a hoop, drilled sensor plate) at the chest front; one-servo claws (MG90S class) on a
  standard wrist plate (4x M3). The claw is fixed mass in simulation, scripted on the real robot.
- Every printed part must pass `tools/strength_check.py` (safety factor >= 3 on real cross-sections)
  and `tools/check_interference.py` (>= 1.0 mm moving clearance, zero overlap) in `hardware/upper_body/`.
- The legs are the original parts, unchanged: nothing the training depends on was touched.
- Plug and play: every module has one standard interface (table in
  `hardware/upper_body/README.md`). Keep those interfaces when changing parts.
- The legs are the original parts. A light restyle (rounding only) is allowed; joint positions, servo
  mounting and foot soles must not change.
- STS3215 on every arm joint in simulation (the only servo with a validated BAM model). The real
  build will run the legs-only policy with scripted arms on cheap servos.
- Legs-only policy with scripted, randomised arm motion. The policy interface is unchanged.
- Design language: like the legs. Servos are the structure, flat plates on their M2 face holes,
  dual-support forks, filleted joints, no rods or trusses, not boxy.
- Clearances for real prints: ≥ 1.0 mm between moving parts (checked on every build). The idler
  bores are 0.3 mm running fits.
- Strength: `tools/strength_check.py` must pass (safety factor ≥ 3 at every critical section).

## Next steps

1. GPU: follow `GPU_RUNBOOK.md` (smoke test first: the Isaac code has never run).
2. If the smoke test fails: fix it on the instance, then commit and push.
3. Export, verify and gate on the VM, including `mujoco_arm_mode: swing` / `random`.
4. Later: adapt `check_actuator.py` / `jitter_probe.py` / `make_fall_bank.py` and the get-up task
   to the arms; write scripted arm gestures for the real robot (`upper_body/motions.py`).

## Things to verify on real hardware before printing

- Horn screw circle `horn_pcd` = 14 mm (assumed): measure your horns.
- The claw servo pocket assumes an MG90S body of 22.8 x 12.2 mm: measure yours (`GR_L`, `GR_W`).
- The M2 face-hole pattern was measured from the reference STEP. Check it against one real servo
  with a test print of `l_elbow_link`, which is small.
