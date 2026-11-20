# Build guide

From a pile of parts to a robot standing on its own. Work through the steps in order.

Bring-up tip: you can do the standing test in step 06 with only the legs, torso and electronics
fitted, and mount the chest, arms and head (step 04) afterwards. A fall during first tests then
can't damage them.

| Step | |
|---|---|
| [01 Printing](01_printing.md) | What to print and the one part that needs care |
| [02 Servo setup](02_servo_setup.md) | IDs, ID map, zero and offset calibration. Read this one carefully |
| [03 Legs](03_legs.md) | Assembling each leg, then the hips and torso |
| [04 Chest, arms and head](04_chest_arms_head.md) | Mounting the chest, both arms, the claws and the head |
| [05 Electronics](05_electronics.md) | Power PCB, wiring, IMU mounting and orientation |
| [06 Software](06_software.md) | Raspberry Pi setup, first power-on, safe standing test |
| [07 Troubleshooting](07_troubleshooting.md) | Common problems and fixes |

Before you start, have everything in [`bom/bom.csv`](../../bom/bom.csv) and the printed parts from
[`bom/printed_parts.csv`](../../bom/printed_parts.csv).
