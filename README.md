# Chhaya

A small 3D-printed humanoid robot: about 0.52 m tall, 2.4 kg, 20 joints. It walks with a
reinforcement-learning policy trained in simulation.

![Chhaya](docs/chhaya.png)

Chhaya stands on the shoulders of [Sir Servington](https://github.com/AsterisCrack/BipedRobot), the
open-source biped by Pablo Gomez Martinez. The legs, electronics, servo model and training stack
are his. Chhaya adds the upper body: two arms with claws, an open chest and a head for sensors.
Credits and licences are in [NOTICE.md](NOTICE.md).

| | |
|---|---|
| Joints | 12 leg (Sir Servington's) + 8 arm: shoulder pitch, shoulder roll, arm yaw, elbow per side |
| Servos | 20x Feetech STS3215 (12 V bus servos), plus 2 micro servos for the claws |
| Hands | one-servo claws on a standard wrist plate (swap for any tool with 4 screws) |
| Head | halo: a face disc in a hoop, with a drilled plate behind it for a camera and small boards |
| Mass | 2.39 kg (upper body 731 g, of which 507 g is servos) |
| Policy | legs only; the arms follow scripted motion, so the policy that walks the 12-servo biped also walks Chhaya |
| Printed parts | 20 for the upper body, every one checked for clearance (≥ 1 mm) and strength (safety factor ≥ 3) |

## Status

| | |
|---|---|
| Upper-body CAD, STL, STEP, URDF | done; passes the interference, clearance and strength checks |
| MuJoCo model and physics checks | done; all pass |
| Isaac Lab integration | written and statically checked, **not yet run** (needs an RTX GPU): start with the smoke test in the GPU runbook |
| Walking policy for Chhaya | not trained yet |
| Physical build | not built yet. Verify the servo hole patterns on one test print first (`hardware/upper_body/README.md`) |

## Layout

```
hardware/                 Sir Servington's hardware: CAD, print files, electronics, robot description
hardware/upper_body/      Chhaya's upper body: CAD source, STL/STEP, check tools, renders, README
training/                 Sir Servington's training code (Isaac Lab, MuJoCo) with upper-body support
training/docs/upper_body/ how the arms are trained, the GPU runbook, the handoff notes
docs/                     pictures and notes for the whole robot
```

## Quick start

```bash
git clone --recurse-submodules https://github.com/<you>/chhaya.git
cd chhaya/training
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest tqdm
(cd robot_motion_reference && python -m gait_studio.bake)
pytest tests/test_arm_motion.py tests/test_upper_body_mujoco.py    # 11 pass, no GPU needed
python -m upper_body.physics_check --out /tmp/phys                 # stands, holds poses, survives a fall
```

- **Print the upper body:** `hardware/upper_body/README.md` (parts, orientations, hardware list, assembly order).
  The STL files are in `hardware/upper_body/cad/out/sts3215/stl/`.
- **Build the legs and electronics:** `hardware/README.md` (the original guide). A cheaper
  electronics option is in [docs/BUDGET_ELECTRONICS.md](docs/BUDGET_ELECTRONICS.md).
- **Train the policy:** `training/docs/upper_body/GPU_RUNBOOK.md`.
- **Change the design:** edit `hardware/upper_body/cad/upper_body.py`, then run
  `hardware/upper_body/regenerate.sh`. It rebuilds every file and refuses to finish if any part
  overlaps, any moving gap is under 1 mm or any safety factor is under 3.

## Design in brief

- **The servos are the structure.** Printed parts are thin plates and forks screwed to the servos'
  own mounting holes, as on the legs. No cosmetic covers.
- **Plug and play.** The head (4 screws on the neck flange), the claws (4 screws on the wrist
  plate), the face disc and every bracket come off without disturbing their neighbours. Hole
  patterns are in `hardware/upper_body/README.md`, so your own parts fit.
- **Checked, not assumed.** Clearance is measured through every joint's full range. Strength is
  calculated on real cross-sections cut from the CAD, against a fall onto a hand, the head and the
  chest.
- **What it can do:** walk, gesture, hold light objects (the claws grip with about 5 N). It is not
  strong enough for housework.

## Licence

`training/` is MIT. `hardware/` is CERN-OHL-P-2.0. Both come from Sir Servington and both
permit this reuse; the new work follows the licence of the directory it is in. See
[NOTICE.md](NOTICE.md).
