# Budget electronics

The original bill of materials is in `hardware/bom/README.md`. These swaps keep the robot able to
walk for less money. **Prices are rough figures for India in late 2026 and were not checked
against shops: confirm before ordering.** None of the swaps has been run on a robot yet.

| Original part | Cheaper option | Rough price | What it costs you |
|---|---|---|---|
| Raspberry Pi 5, 2 GB | Raspberry Pi 5, 1 GB | ₹4,000–5,000 | nothing: the policy needs little memory. A Pi 4 is untested and has a different UART setup |
| Custom power PCB | 5 V 5 A step-down module (XL4015 type), inline fuse, switch, XT60 lead; servos wired straight to the battery | ₹400–600 | you solder a small harness |
| BNO055 IMU | MPU6050 | ₹150–250 | the BNO055 estimates tilt on-chip; the MPU6050 needs a small filter added to the robot software |
| Waveshare bus servo adapter | keep it | ₹500–700 | already cheap; home-made wiring is less reliable at 1 Mbaud |
| 3S LiPo 2200 mAh | keep it, or a 3S high-drain 18650 pack with a protection board | ₹1,200–2,000 | the 18650 pack is safer but heavier |
| STS3215 servos | keep them | — | the simulation's servo model exists only for them |
| Claw servos (Chhaya only) | MG90S-class micro servo, 2 pieces | ₹150–250 each | driven from the Pi (PWM), scripted like the arms |

The servos are most of the cost. Everything else comes to roughly ₹7,000–9,000 with these swaps.
