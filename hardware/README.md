# Chhaya hardware

**Open hardware for a small 20-servo humanoid that learns to walk in simulation.**
CAD, print files, bill of materials, power PCB, robot description (URDF) and build guide.

<p align="center">
  <img src="media/chhaya/overview.png" alt="Chhaya" width="90%"/>
</p>

The picture is a render of the model: Chhaya has not been built yet.

---

## Specs

| | |
|---|---|
| Height | About 54 cm standing |
| Mass | 2.42 kg in the model (with battery, Raspberry Pi 5, wiring and electronics) |
| Degrees of freedom | 20: per leg hip yaw, hip roll, hip pitch, knee, ankle roll, ankle pitch; per arm shoulder pitch, shoulder roll, arm yaw, elbow |
| Actuators | 20× Waveshare ST3215 serial bus servos (12 V, 30 kg·cm), one shared bus at 1 Mbaud; 2 micro servos for the claws |
| Hands | One-servo claws on a standard wrist plate |
| Head | Halo head with a drilled plate for a camera and small boards |
| Compute | Raspberry Pi 5 on the electronics rack under the chest, 50 Hz control loop. |
| IMU | BNO055 on I2C |
| Power | 3S LiPo (11.1 V, 2200 mAh) through a custom power PCB |
| Structure | 34 printed parts, PLA or PETG |
| Cost | ≈ ₹58,700 with a Raspberry Pi 5 (2 GB). Cheaper options in [`bom/`](bom/README.md#cheaper-options) |

---

## Where to start

| I want to... | Go to |
|---|---|
| See what to buy and what it costs | [`bom/`](bom/) |
| Print the parts | [`print/`](print/) |
| Understand or modify the design | [`docs/design.md`](docs/design.md), [`cad/`](cad/) (STEP for any CAD tool; parametric source for the chest, head and arms) |
| Order or build the power board | [`electronics/power_pcb/`](electronics/power_pcb/) |
| Wire it | [`electronics/wiring/`](electronics/wiring/) |
| Build it | [`docs/build/`](docs/build/) |
| Simulate it | [`robot_description/`](robot_description/) (URDF + meshes) |

---

## Cost

| Part | ₹ |
|---|---|
| 20× Waveshare ST3215 servos | 41,400 |
| Raspberry Pi 5, 2 GB, Active Cooler and 12 V to 5 V step-down | 9,630 |
| 2× micro servos for the claws (estimate) | 560 |
| BNO055 IMU | 1,600 |
| Power PCB | 2,350 |
| Waveshare Bus Servo Adapter (A) | 470 |
| 3S 2200 mAh LiPo | 1,880 |
| Fuses, power switch, servo leads (estimate) | 820 |
| **Total** | **≈ 58,700** |

List prices converted at ₹94 per US dollar; details, links and cheaper options are in [`bom/`](bom/). The policy
is a small network that runs on the CPU, so a Raspberry Pi 5 is enough. Not in the prices: the microSD card,
about 0.75 kg of filament, wires and screws. Tools: a 3D printer, a soldering iron and a LiPo balance charger.

---

## Repository layout

```
cad/
├── step/              One STEP per printed part; the legs-and-torso assembly (zipped)
└── parametric/        CadQuery source of the chest, head, arms and claws + check results
print/stl/             One STL per printed part, with print notes in print/README.md
robot_description/     URDF + meshes: chhaya.urdf is the whole robot
electronics/
├── Schematic_RobotPCB.pdf
├── power_pcb/         Power PCB: EasyEDA source, gerbers, BOM, pick-and-place, render
└── wiring/            Servo ID map, Raspberry Pi pinout, claw servos, IMU mounting
bom/                   Purchased parts and printed parts (CSV)
docs/design.md         Mechanical design: interfaces, strength, clearances, kinematics
docs/build/            Build guide
tools/, regenerate.sh  Rebuild and check everything generated from the parametric CAD
media/                 Photos and renders
```

The training code reads the robot description from here through `training/robot_hardware`.

---

## Credits

<p>
  <img src="media/pcbway.png" alt="PCBWay logo" height="40"/>
</p>

- The power PCB was fabricated and assembled by [PCBWay](https://www.pcbway.com/), who sponsored the board.
- The servo model used in simulation builds on Rhoban's [BAM](https://github.com/Rhoban/bam).
- The URDF was exported from Fusion 360 with [fusion2urdf](https://github.com/syuntoku14/fusion2urdf)
  (MIT, see [`robot_description/LICENSE-fusion2urdf`](robot_description/LICENSE-fusion2urdf)).

---

## License

The hardware design (CAD, print files, PCB, robot description, documentation) is licensed under the
[CERN Open Hardware Licence Version 2 - Permissive](LICENSE) (CERN-OHL-P-2.0).
The exporter template files in `robot_description/` keep their own MIT licence; the robot design itself is
covered by CERN-OHL-P. The software in `../training/` is MIT.

Modification notice (CERN-OHL-P-2.0 §3.3): the legs, hips, torso and power PCB are modified from
[BipedRobotHardware](https://github.com/AsterisCrack/BipedRobotHardware) by Pablo Gomez Martinez. In November 2026
the chest, head, arms, claws and electronics rack were added (parametric CAD, STEP, STL, URDF links and meshes),
the documentation was rewritten for the whole robot, and the build and check tools were added.
