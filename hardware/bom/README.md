# Bill of materials

| File | |
|---|---|
| [`bom.csv`](bom.csv) | Everything you buy, with prices and links |
| [`printed_parts.csv`](printed_parts.csv) | Everything you print: 14 parts for the legs and torso, 20 for the chest, head, arms, claws and rack |

## Summary

| Qty | Part | USD |
|---|---|---|
| 20 | Waveshare ST3215 servo, 12 V (12 legs + 8 arms) | 439.80 |
| 1 | Raspberry Pi 5, 2 GB | 87.50 |
| 1 | 12 V to 5 V 5 A step-down, USB-C output (estimate) | 10.00 |
| 2 | Micro servo for the claws, MG90S class (estimate) | 6.00 |
| 1 | BNO055 IMU breakout | 17.00 |
| 1 | Power distribution PCB ([`electronics/power_pcb`](../electronics/power_pcb/)) | 25.00 |
| 1 | Waveshare Bus Servo Adapter (A) | 4.99 |
| 1 | 3S LiPo, 2200 mAh, 25C or more, XT60 | 20.00 |
| 2 | Blade fuses, 10 A (estimate) | 2.00 |
| 1 | Power switch, 15 A at 12 V DC (estimate) | 3.00 |
| 4 | Servo extension leads, 30 cm (estimate) | 4.00 |
| | **Total** | **≈ $619** |

A 1 GB Raspberry Pi 5 ($55) brings it to about $587: the policy is a small network that needs little memory.
A Jetson Orin Nano does not fit on the electronics rack under the chest
([`docs/ELECTRONICS_LAYOUT.md`](../../docs/ELECTRONICS_LAYOUT.md)), so the Pi 5 is the computer.

The servos are most of the cost: the 8 arm servos add about $176 to the legs-only robot.

Prices were checked in October 2026 (Raspberry Pi list prices as of 12 Oct 2026) and drift; check before
you order. Rows marked "estimate" are typical prices for generic parts rather than a specific listing.

**Not in the prices:** the microSD card, about 0.75 kg of PLA, wires, the XT60 and barrel leads, M3 heat-set
inserts and the M2/M3 screws for the arms (listed under "fasteners" in the CSV; the legs use only the screws
that come with the servos). You also need a 3D printer, a soldering iron and a LiPo balance charger.

**Power budget.** Each PCB input passes one half of its SB2060FCT diode, rated 10 A, hence the 10 A fuses.
Walking draws an estimated 5-7 A for all 20 servos (the hips hold about 1.2 N·m, about 0.5 A each). If a fuse
blows in normal walking, the diode is the limit: replace it with a part rated 20 A per half (same package)
before fitting larger fuses.

## Setting up the Raspberry Pi

The policy is a small network that the runtime runs on the CPU with onnxruntime, well within a Raspberry Pi 5's
budget at 50 Hz. The legs-only reference robot used a Jetson; the wiring is the same.

- **Power:** the Pi 5 needs 5 V over USB-C, not battery voltage. Connect the 12 V to 5 V step-down to the
  PCB's computer terminal (labelled JETSON on the board) and its USB-C lead to the Pi.
- **Wiring:** the Pi's 40-pin header uses the same pins as the Jetson: UART on pins 8/10 to the servo
  adapter, I2C on pins 3/5 to the BNO055.
- **Software:** [BipedRobotJetson](https://github.com/AsterisCrack/BipedRobotJetson) runs as-is with two
  settings changed. Set `UART_PORT=/dev/ttyAMA0` (after enabling the UART with `dtparam=uart0=on`) and set
  `i2c_bus: 1` in `config/hardware.yaml`. Skip `Jetson.GPIO` when installing the hardware requirements.

The reference robot runs on the Jetson; the Raspberry Pi setup has not been tested on it yet.
