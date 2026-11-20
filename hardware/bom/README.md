# Bill of materials

| File | |
|---|---|
| [`bom.csv`](bom.csv) | Everything you buy, with prices and links |
| [`printed_parts.csv`](printed_parts.csv) | Everything you print: 14 parts for the legs and torso, 20 for the chest, head, arms, claws and rack |

## Summary

| Qty | Part | ₹ |
|---|---|---|
| 20 | Waveshare ST3215 servo, 12 V (12 legs + 8 arms) | 41,400 |
| 1 | Raspberry Pi 5, 2 GB | 8,220 |
| 1 | Raspberry Pi 5 Active Cooler | 470 |
| 1 | 12 V to 5 V 5 A step-down, screw-terminal output (estimate) | 940 |
| 1 | USB-C power pigtail to the Pi (estimate) | 150 |
| 2 | Micro servo for the claws, MG90S class (estimate) | 560 |
| 1 | BNO055 IMU breakout | 1,600 |
| 1 | Power distribution PCB ([`electronics/power_pcb`](../electronics/power_pcb/)) | 2,350 |
| 1 | Waveshare Bus Servo Adapter (A) | 470 |
| 1 | 3S LiPo, 2200 mAh, 25C or more, XT60 | 1,880 |
| 2 | Blade fuses, 10 A (estimate) | 180 |
| 1 | Power switch, 15 A at 12 V DC (estimate) | 280 |
| 4 | Servo extension leads, 30 cm (estimate) | 360 |
| | **Total** | **≈ ₹58,860** |

The servos are most of the cost. Prices are the suppliers' list prices from October 2026, converted at ₹94 per
US dollar (November 2026) and rounded to ₹10; Indian shops, shipping and import duty change them, so check before
you order. Rows marked "estimate" are typical prices for generic parts rather than a specific listing.

**Not in the prices:** the microSD card, about 0.75 kg of PLA, wires, the XT60 and barrel leads, the screws,
inserts and battery strap (listed under "fasteners" in the CSV; the legs use only the screws
that come with the servos). You also need a 3D printer, a soldering iron and a LiPo balance charger.

## Cheaper options

Rough Indian prices, late 2026, not checked against shops. None of these swaps has been run on a robot yet.

| Part | Cheaper option | Rough price | What it costs you |
|---|---|---|---|
| Raspberry Pi 5, 2 GB | Raspberry Pi 5, 1 GB | ₹4,000–5,000 | nothing: the policy needs little memory |
| Power PCB | 5 V 5 A step-down module (XL4015 type), inline fuse, switch, XT60 lead; servos wired straight to the battery | ₹400–600 | you solder a small harness |
| BNO055 IMU | MPU6050 | ₹150–250 | the BNO055 estimates tilt on-chip; the MPU6050 needs a small filter in the robot software |
| 3S LiPo 2200 mAh | a 3S high-drain 18650 pack with a protection board | ₹1,200–2,000 | safer but heavier |
| ST3215 servos | keep them | | the simulation's servo model is fitted to them. Hobby servos: [`docs/CHHAYA_LITE.md`](../../docs/CHHAYA_LITE.md) |

**Power budget.** Each PCB input passes one half of its SB2060FCT diode, rated 10 A, hence the 10 A fuses.
Walking draws an estimated 5-7 A for all 20 servos (the hips hold about 1.2 N·m, about 0.5 A each). If a fuse
blows in normal walking, the diode is the limit: replace it with a part rated 20 A per half (same package)
before fitting larger fuses.

Setting up the Raspberry Pi: [build guide step 06](../docs/build/06_software.md).
