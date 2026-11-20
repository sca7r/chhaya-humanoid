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
| 1 | Raspberry Pi 5 Active Cooler | 5.00 |
| 1 | 12 V to 5 V 5 A step-down, USB-C output (estimate) | 10.00 |
| 2 | Micro servo for the claws, MG90S class (estimate) | 6.00 |
| 1 | BNO055 IMU breakout | 17.00 |
| 1 | Power distribution PCB ([`electronics/power_pcb`](../electronics/power_pcb/)) | 25.00 |
| 1 | Waveshare Bus Servo Adapter (A) | 4.99 |
| 1 | 3S LiPo, 2200 mAh, 25C or more, XT60 | 20.00 |
| 2 | Blade fuses, 10 A (estimate) | 2.00 |
| 1 | Power switch, 15 A at 12 V DC (estimate) | 3.00 |
| 4 | Servo extension leads, 30 cm (estimate) | 4.00 |
| | **Total** | **≈ $624** |

A 1 GB Raspberry Pi 5 ($55) brings it to about $592: the policy is a small network that needs little memory.
A Jetson Orin Nano does not fit on the electronics rack under the chest
([`docs/ELECTRONICS_LAYOUT.md`](../../docs/ELECTRONICS_LAYOUT.md)), so the Pi 5 is the computer.

The servos are most of the cost: the 8 arm servos add about $176 to the legs-only robot.

Prices were checked in October 2026 (Raspberry Pi list prices as of 12 Oct 2026) and drift; check before
you order. Rows marked "estimate" are typical prices for generic parts rather than a specific listing.

**Not in the prices:** the microSD card, about 0.75 kg of PLA, wires, the XT60 and barrel leads, the screws,
inserts and battery strap (listed under "fasteners" in the CSV; the legs use only the screws
that come with the servos). You also need a 3D printer, a soldering iron and a LiPo balance charger.

**Power budget.** Each PCB input passes one half of its SB2060FCT diode, rated 10 A, hence the 10 A fuses.
Walking draws an estimated 5-7 A for all 20 servos (the hips hold about 1.2 N·m, about 0.5 A each). If a fuse
blows in normal walking, the diode is the limit: replace it with a part rated 20 A per half (same package)
before fitting larger fuses.

Setting up the Raspberry Pi: [build guide step 06](../docs/build/06_software.md).
