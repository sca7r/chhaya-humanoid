# Chhaya

[![Live demo](https://img.shields.io/badge/live%20demo-try%20it%20in%20your%20browser-0071e3.svg)](https://sca7r.github.io/chhaya-humanoid/)
[![Software: MIT](https://img.shields.io/badge/software-MIT-blue.svg)](training/LICENSE.txt)
[![Hardware: CERN-OHL-P-2.0](https://img.shields.io/badge/hardware-CERN--OHL--P--2.0-green.svg)](hardware/LICENSE)

**Chhaya is a small humanoid you can print at home.** It stands 54 cm tall, moves on 20 servos, grips with
two claws and carries sensors in a halo-shaped head. It learned to walk on its own, in simulation, by trial
and error: one neural network drives all 20 of its joints, 50 times a second.

*Chhaya* is Hindi for shadow. Tap the floor and it follows you around.

<p align="center">
  <img src="docs/media/skills.gif" alt="Chhaya walking forward, stepping sideways, turning and recovering from pushes" width="49%"/>
  <img src="docs/media/crowd.gif" alt="Sixteen copies of Chhaya walking with random commands" width="49%"/>
</p>
<p align="center">
  <img src="docs/media/four_views.png" alt="Chhaya mid-stride from the front, side, back and an isometric view" width="90%"/>
</p>

**[Walk Chhaya in your browser](https://sca7r.github.io/chhaya-humanoid/)**: the same brain, running on your
device, with nothing to install.

## What Chhaya can do

- **Walk anywhere you point.** Forward, backward and sideways, turning on the spot or while walking. Tap a
  spot and it turns, walks there and stops.
- **Keep its balance.** Shove it and it steps to catch itself. Its arms swing opposite to its legs, the way
  yours do, which halved its falls in testing.
- **Walk evenly.** Its brain is mirror-symmetric, so its left and right legs move as mirror images: no limp.
- **Cope with the real world.** It trained with random changes to its weight, its servos and the floor, so it
  still walks 300 g lighter or 150 g heavier than designed.
- **Grip light things.** Each claw is one micro servo, with about 5 N of grip.
- **Change hands and heads.** The claws, the head and the face disc each come off with 4 screws, and their
  hole patterns are documented, so your own parts fit.

## How Chhaya learned to walk

Chhaya practised in [Isaac Lab](docs/ISAACLAB3.md), a GPU physics simulator, as a thousand copies at once.
The trainer, [SuperMimic](docs/SUPERMIMIC.md), learns how Chhaya's body moves and then improves the walk
by imagining the outcome of each step. A night on a laptop RTX 3060 was enough to go from falling to walking.

Along the way it learned to stop limping (a mirror-symmetric brain and a reference gait), to swing its arms for
balance, and to recover from pushes. Every version is checked in a second simulator, MuJoCo, before it is
trusted. The full story, with videos of each stage, is in [`docs/JOURNEY.md`](docs/JOURNEY.md).

## What Chhaya is made of

| Part | Quantity | Notes |
|---|---|---|
| Waveshare ST3215 serial bus servo, 12 V, 30 kg·cm | 20 | 12 in the legs, 8 in the arms, one shared bus |
| MG90S micro servo | 2 | one per claw |
| Raspberry Pi 5 | 1 | its computer, on a rack under the chest |
| BNO055 IMU | 1 | its sense of balance |
| Waveshare Bus Servo Adapter (A) | 1 | connects the Pi to the servo bus |
| Power distribution PCB | 1 | custom, 55 x 50 mm, design files in [`hardware/electronics/`](hardware/electronics/) |
| 3S LiPo, 11.1 V, 2200 mAh, XT60 | 1 | roughly an hour of walking (estimate) |
| Printed parts, PLA or PETG | 34 | [`hardware/print/`](hardware/print/); the servos themselves are the frame |

The full bill of materials, with suppliers and prices, is in [`hardware/bom/`](hardware/bom/). Chhaya costs
about ₹58,860 with a Raspberry Pi 5, most of it the 20 servos. Cheaper options are in the
[BOM](hardware/bom/README.md#cheaper-options).

| Specification | Value |
|---|---|
| Height | about 54 cm |
| Mass | 2.42 kg in the model, with battery and Raspberry Pi 5 |
| Joints | per leg: hip yaw, hip roll, hip pitch, knee, ankle roll, ankle pitch. Per arm: shoulder pitch, shoulder roll, arm yaw, elbow |
| Control | 50 Hz; the claws are scripted |
| Walking speed | up to 0.2 m/s forward and backward, 0.15 m/s sideways, turning in place |

Every printed part passes automatic overlap, clearance (1 mm or more) and strength (safety factor 3 or more)
checks, rerun whenever a dimension changes.

## Getting started

1. **Source the hardware.** Order the parts in the [bill of materials](hardware/bom/).
2. **Print the parts.** STL files and settings are in [`hardware/print/`](hardware/print/).
3. **Assemble the robot.** Follow the [build guide](hardware/docs/build/), from servo setup to the
   electronics.
4. **Try it in simulation.** No robot or GPU is needed. See [Run in simulation](#run-in-simulation).
5. **Train or fine-tune a policy.** Any recent NVIDIA GPU works, a laptop RTX 3060 included. See
   [Teach Chhaya yourself](#teach-chhaya-yourself).
6. **Deploy to the robot.** The Raspberry Pi runtime is next (see [Where Chhaya is today](#where-chhaya-is-today)).

## Run in simulation

The simulation stack runs on Linux with Python 3.11. Everything runs from inside `training/`.

```bash
git clone --recurse-submodules https://github.com/sca7r/chhaya-humanoid.git
cd chhaya-humanoid/training
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest tqdm

# bake the reference gait the policy imitates (one time)
(cd robot_motion_reference && python -m gait_studio.bake --style processed/gait_table_style.yaml --out processed/gait_table.npz)

# check the robot model: stands, holds poses, survives a fall
python -m chhaya.physics_check --out /tmp/phys

# run the test suite
pytest tests
```

To try a trained policy without a GPU, download `chhaya-policies-v0.1.0.zip` from the
[release](https://github.com/sca7r/chhaya-humanoid/releases), unzip it into `training/exported/` and run the MuJoCo
check. It walks the robot through standing, walking and control-delay tests and prints a pass or fail report:

```bash
python -m src.mujoco.policy_gate --bundle exported/policies/chhaya
```

## Talking to Chhaya

Chhaya understands one command: how fast to go forward, how fast to go sideways and how fast to turn.
Every way of driving it comes down to producing that command, so a new way of talking to it needs no
retraining.

| Input | How it works | Status |
|---|---|---|
| Tap a spot | tap the floor: Chhaya turns, walks there and stops | available in the browser demo |
| Keyboard and buttons | arrow keys walk and turn, `,` `.` step sideways, `Space` stops; the same on on-screen buttons | available in the browser demo |

The browser demo is `web/`: MuJoCo compiled to WebAssembly, the policy in onnxruntime-web and the same servo
model, observation and action map as the MuJoCo check (`web/sim.mjs`). To refresh it with a new policy:

```bash
cd training
python -m src.mujoco.export_web --bundle exported/policies/chhaya --trace 300   # writes web/assets/
cd ../web && npm install --no-save @mujoco/mujoco@3.17.0 onnxruntime-web@1.20.1 && node test_sim.mjs .
python3 -m http.server 8000                                                 # open http://localhost:8000
```

The test checks the observations against the Python environment step for step, then lets the policy walk and
reach four tapped spots. Pushing to `main` publishes `web/` (`.github/workflows/pages.yml`).

Commands are limited to the range the policy was trained on (`config/chhaya.yaml`), whatever the input.

## Teach Chhaya yourself

Training uses [Isaac Lab 3.0](docs/ISAACLAB3.md) on the Newton physics backend, without Isaac Sim. Install
it as described there, then:

```bash
cd training
# train (the laptop preset fits a 6 GB GPU; --minutes is a wall-clock budget)
python src/isaaclab/train_supermimic.py --model eth --config_path config/chhaya.yaml \
    --num_envs 1024 --preset laptop --minutes 300 --save_every 500

# evaluate a checkpoint on a fixed command (vx m/s, vy m/s, wz rad/s) and record it
python src/isaaclab/train_supermimic.py --model eth --config_path config/chhaya.yaml --num_envs 64 \
    --eval runs/<run>/supermimic_last.pt --command 0.15 0 0 --record videos/walk.npz

# render the recording as a video
MUJOCO_GL=egl python -m src.mujoco.replay_video videos/walk.npz --iso --out videos/walk.mp4

# export the deployment bundle (ONNX) and check it in MuJoCo
python src/isaaclab/train_supermimic.py --model eth --config_path config/chhaya.yaml --num_envs 4 \
    --export runs/<run>/supermimic_last.pt --out_dir exported/chhaya
python -m src.mujoco.policy_gate --bundle exported/chhaya
```

The trainer is [SuperMimic](docs/SUPERMIMIC.md): it learns a model of the robot's dynamics and improves the
policy through imagined rollouts. A standard SAC trainer is also included ([`docs/TRAINING.md`](docs/TRAINING.md)).
To train on a rented GPU, see [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md).

## Project structure

```
chhaya/
├── hardware/                  everything physical
│   ├── cad/                   parametric CAD (chhaya_cad.py) and the checks it must pass
│   ├── print/                 STL files to print
│   ├── bom/                   bill of materials
│   ├── electronics/           power PCB
│   ├── robot_description/     URDF and meshes
│   ├── docs/build/            step-by-step build guide
│   └── regenerate.sh          rebuilds every generated part and reruns the checks
├── training/                  reinforcement learning, run everything from here
│   ├── config/                robot and training configs (chhaya.yaml)
│   ├── envs/                  Isaac Lab and MuJoCo environments, robot models
│   ├── supermimic/            model-based trainer
│   ├── actuator/              servo model fitted to the STS3215
│   ├── src/isaaclab/          training, evaluation, ONNX export
│   ├── src/mujoco/            MuJoCo checks, playback and video rendering
│   └── tests/                 test suite
├── web/                       browser demo (GitHub Pages): tap-to-walk, physics and policy in the browser
└── docs/                      training guides and results
```

## Where Chhaya is today

| Area | State |
|---|---|
| CAD, print files, URDF | done, all automatic checks pass |
| Simulation model | done, physics checks pass ([`docs/physics/`](docs/physics/)) |
| Walking | learned in simulation ([what it can do](#what-chhaya-can-do)), checked in MuJoCo |
| Turning | turns at about 60% of the asked-for rate on the spot. Being improved |
| Hip servos | work hard while walking (about 1.15 N·m RMS, above the continuous rating). Being reduced |
| Getting up | a get-up policy for the whole robot (arms included) is being trained for the next release |
| Physical build | Chhaya has not been built yet |
| On-robot software | the runtime for the Raspberry Pi 5 (20 servos, IMU, claws) is next |

## Contributing

Issues and pull requests are welcome. Before opening a pull request:

- run `pytest tests` and the physics check from `training/` ([Run in simulation](#run-in-simulation));
- change the CAD in `hardware/cad/parametric/chhaya_cad.py` and run `hardware/regenerate.sh`, rather than
  editing the generated URDF, MuJoCo, USD or STL files;
- keep the leg geometry and servo mounting as they are: the trained policies depend on them.

## License

The software in `training/` is released under the [MIT License](training/LICENSE.txt). The hardware in
`hardware/` is released under the [CERN Open Hardware Licence Version 2, Permissive](hardware/LICENSE). Chhaya
builds on Pablo Gomez Martinez's open-source biped (MIT software, CERN-OHL-P-2.0 hardware); the copyright
notice is in `training/LICENSE.txt` and the hardware modification notice in [`hardware/README.md`](hardware/README.md#license).
