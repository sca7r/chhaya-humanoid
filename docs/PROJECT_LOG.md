# Project log: decisions and why (for a collaborator picking up the project)

A condensed record of the design conversation with Harsh (2026-11-09 to 2026-11-18), so the reasons
behind the current state are not lost. It replaces reading the chat itself, which is long and full of
plans that were later changed. Newest first. `HANDOFF.md` has the current state and next steps.

## 2026-11-19: SuperMimic rewritten the ETH way; training on the RTX 3060 laptop

- Cross-checked against the paper, the official code (released 2026-11-10) and the awesome-supermimic list.
  The official code and the only velocity-walking ports (ETH terrain-grounded imagination, XSoma) learn only
  the dynamics and compute reward/termination analytically. Harsh chose that (ETH has hardware proof):
  `supermimic/eth.py`, default `--model eth`; `v1` kept for comparison. Details: `SUPERMIMIC.md`.
- Laptop: Isaac Lab 3.0 (Newton, no Isaac Sim) runs on the RTX 3060 6 GB in WSL2; Chhaya env at 1024
  robots ≈ 1,800 robot-steps/s. Runs go through a job queue (`~/queue` in WSL, Windows task `ChhayaQueue`).
  Every laptop change is logged in WSL `~/CHHAYA_RESTORE.md`: Harsh wants the laptop back to default after.

## 2026-11-18: SuperMimic trainer (optional)

- Asked to use SuperMimic (arXiv:2610.28480: learned world model + backpropagation through imagined
  rollouts). No code had been published, so it was reimplemented in `training/supermimic/`, as a
  second trainer next to SAC. SAC remains the default. Details and results: `SUPERMIMIC.md`.
- It learns a small test task (score 0.08 → 0.98). On Chhaya, CPU, 4 robots: stable, but the policy
  ended worse than standing still, most likely too little data per update (10 transitions per
  iteration vs ~10,000 in the paper). A floor on new data per iteration was added. **Open: does it
  learn to walk on a GPU?**
- Packaging: `training/robot_hardware` is a symlink; zips are built with `zip -y` so it stays one.

## 2026-11-17: RoboKit reviewed, not adopted

- github.com/openrobokit/robokit (MIT, MuJoCo Warp, PPO/FastSAC examples). No servo model, delay or
  sensor model; examples are G1/H1/Go1. Not a reason to switch. Worth borrowing later: its FastSAC,
  coarse-to-fine physics training, and human-motion retargeting (for future motions).

## 2026-11-16: test suite fixed; Lite servo data checked

- The "7 known failures" were mostly self-inflicted: the gait table had been baked without
  `--style processed/gait_table_style.yaml`. Two tests had stale expectations. All tests pass now.
  `chhaya.yaml` does not use the gait table at all.
- Lite servo files checked against datasheets: RDS3225 no-load speed corrected (0.17 s/60°, not 0.13).
  RDS3225 torque differs between sources (25 / 28 / 34.5 kgf·cm): the model uses 25.
- Finding: Lite's risk is gear play, not torque. With ≥ 2° of play the robot shakes itself over
  without a policy. Legs need the 270° RDS3225; shoulder pitch (229° range) exceeds the MG996R.
- An "actuator network" (NN for servo noise) needs recordings from real servos; not from datasheets.

## 2026-11-15: Chhaya Lite (budget track)

- Harsh asked for a cheap build with hobby servos: RDS3225 legs, MG996R arms, MG90S claws, a wire
  soldered to each servo's pot for feedback. Simulated as `config/chhaya_lite.yaml` (servo files are
  datasheet estimates). Body and frame unchanged until training says these servos are worth it.
- Answered: PWM hobby servos cannot be used as-is on policy joints (no feedback, unknown controller);
  pot tapping gives feedback with ~0.5–1° noise. Suggested budget mix: STS3215 legs + MG996R arms.

## 2026-11-14: Isaac Lab 3.0 (Newton backend, no Isaac Sim)

- Moved training from Isaac Lab 2.x / Isaac Sim (PhysX, needs RT cores, ≥ 8 GB) to Isaac Lab 3.0
  kit-less with the Newton / MuJoCo-Warp solver, so it can run on Harsh's RTX 3060 laptop (6 GB).
  Tensors and (w, x, y, z) quaternions are kept inside the env; `envs/isaaclab/compat.py` converts.
  Tested on a CPU only. `ISAACLAB3.md`.
- Expected results "similar, not identical" to Isaac Sim: different contact solver; reward weights
  were tuned on PhysX. There was never a Chhaya (20-joint) policy on either simulator.

## 2026-11-12/13: whole-body policy, electronics, servo load, naming

- **Whole-body policy:** Harsh does not want scripted arms. One policy for all 20 joints
  (action 12 legs + 8 arms, 74-value observation frame). Scripted arms kept only as a fallback.
- **Electronics on board:** Raspberry Pi 5 + 3S 2200 mAh battery on a printed rack under the chest
  (chest raised 23 mm); custom PCB and bus adapter in the torso; a Jetson no longer fits. A 12 V tether
  into the PCB is for the bench only. `ELECTRONICS_LAYOUT.md`.
- **Servos:** STS3215 kept on all 20 joints (worst static load 46 % of stall). DS3225-class PWM servos
  were considered and rejected for the policy joints.
- **Naming:** Chhaya is presented as its own robot. Credit to the open-source biped it derives from is
  kept only where the licences require it (`NOTICE.md`, LICENSE files).
- GPU options compared: RTX 4090 rental ≈ $0.34/h; JarvisLabs offers H200/A100/L4/A30; with Isaac Lab
  3.0 no RT cores are needed, so all of those work. Our SAC uses one GPU. Storage ≈ 30 GB.

## 2026-11-09/11: the robot

- 20 servo joints: 12 legs (unchanged from the base biped: joint positions, servo mounts, soles fixed)
  + 4 per arm (pitch, roll, yaw, elbow). No waist. Open chest, halo head (H3), one-servo claws on a
  standard wrist plate. Plug-and-play module interfaces. Every part: ≥ 1 mm moving clearance, safety
  factor ≥ 3, lightweight. One robot, one project: no "upper body" side project.

## How Harsh works with this

- Deliveries as a plain `.zip` of the repo (a git `.bundle` did not open for him).
- Training hardware: RTX 3060 laptop (6 GB) first; rented GPUs (RunPod, JarvisLabs) if needed.
- Wants plain explanations of what was tested and what is still unknown.

## Open items (also in HANDOFF.md)

1. GPU smoke test (`GPU_RUNBOOK.md`), then a first SAC run on `chhaya.yaml`.
2. SuperMimic on the GPU, compared with SAC at equal wall time.
3. `export_onnx.py`, `policy_gate.py`, `play.py` for 20 joints and for SuperMimic checkpoints.
4. On-robot runtime: add the 8 arm servos and the claws.
5. `hardware/print/stl/body.stl` is missing (only STEP); real part masses not measured.
6. Lite: measure one RDS3225 and one MG996R (gear play first), then decide on the frame redesign.
