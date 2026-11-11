# Where the electronics go (measured 2026-11-12)

Short answer: **not everything fits yet.** The chest frame sits where the original robot carries
its computer and battery (on top of the torso shelf), and no mount for them has been designed for
Chhaya. The spaces below were measured on the CAD with exact collision tests at the zero pose
(robot frame: x forward, y left, z up, mm, origin at the torso).

## Free spaces

| Space | Free box (x × y × z, mm) | Access |
|---|---|---|
| A. Chest centre, between the two shoulder servos | 47 × 50 × 26 | front and back windows, 42 × 21 |
| B. Shelf rear, under the chest | 19 × 120 × 32 | open at both sides |
| C. Torso cavity, under the shelf | 82 × 32 × 28 | from below / the ends |
| D. Behind the chest back plate ("backpack") | unlimited depth, 130 wide, from z = 96 up | open |
| E. Head sensor plate | M2.5 holes on a 10 mm grid inside the 72 mm halo | open |

## What fits where

| Part | Size (mm) | Fits in | Verdict |
|---|---|---|---|
| IMU (BNO055 breakout) | 20 × 27 × 4 | its `IMU_tray` on the torso, as on the original | fits, unchanged |
| Servo bus adapter (Waveshare A) | ≈ 42 × 33 × 12 | A (chest centre) or C | fits |
| 12 V → 5 V step-down (for a Pi) | ≈ 45 × 25 × 15 | A or C | fits |
| Camera / small sensors | | E (head plate) | fits |
| Power PCB | 55 × 50 × ≈ 15 | none of A–C (too wide for each) | **needs D, or a narrower board** |
| Raspberry Pi 5 | 85 × 56 × 17 (≈ 30 with cooler) | none of A–C | **needs D** |
| Jetson Orin Nano dev kit | ≈ 100 × 79 × 35 | none of A–C | **needs D; heavy for a backpack** |
| 3S 2200 mAh LiPo | ≈ 105 × 34 × 24, ≈ 180 g | none of A–C | **needs D, or a different pack** |

Sizes of the adapter, step-down and battery are typical values, not measured parts: check yours.

## What has to be designed

1. **A backpack module** on the chest back plate (4 × M3, like the other plug-and-play modules):
   a plate that carries the computer upright and the power PCB, with the battery in a cradle at
   the bottom, as low and as close to the back plate as possible.
2. **Or raise the chest about 35–40 mm** to open a full bay between the shelf and the chest. This
   keeps the weight centred but makes the robot taller and moves the arms up.
3. A smaller battery helps either way: the standard 2200 mAh pack is the hardest part to place.

## Why this matters for training

The simulated torso (`base_link`, 0.633 kg) still has the original robot's mass distribution,
with the computer and battery on top of the shelf, which is now inside the chest. Wherever they
end up, the model's masses must be updated to match and the URDF / MJCF / USD regenerated
**before** the GPU run. A battery 4–5 cm further back than the model thinks is the kind of error
a walking policy does not survive on the real robot.
