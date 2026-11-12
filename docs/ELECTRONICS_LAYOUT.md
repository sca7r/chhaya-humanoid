# Electronics: where everything goes (updated 2026-11-12)

Short answer: **everything fits now.** The chest was raised 23 mm and stands on a new printed
electronics rack. The battery and a Raspberry Pi 5 sit under the chest, low and centred over the hips,
where the original robot carried its computer and battery. Full design notes are in
`hardware/docs/design.md` (section Electronics). The fit is checked on every rebuild by
`hardware/tools/check_interference.py`.

## How the original robot (Sir Servington) does it

Seen in its photo (`hardware/media/robot.jpeg`) and measured on its body CAD:
- The top of the body is a rim tray with 4 bosses for the Jetson Orin Nano dev kit (hole pattern 58 x 86 mm).
- The battery is strapped on top of the Jetson, with a cosmetic "top hat" over it.
- The power PCB and the servo bus adapter hang inside the torso, beside its centre plate.
- The IMU sits in its own tray inside the torso.

## Chhaya

| Where | What | Clearance |
|---|---|---|
| Inside the torso (unchanged) | power PCB, servo bus adapter, IMU | as on the original |
| Rack plate, in the rim tray | 3S 2200 mAh LiPo, lengthwise, strapped | 1.5 mm to the body, 17 mm to the chest |
| Rack deck, above the battery | Raspberry Pi 5 with active cooler, ports out the back | 1.7 mm to the chest |
| Inside the chest box | 12 V → 5 V step-down for the Pi, wiring | 47 x 50 x 26 mm free |
| Head plate | camera, small sensors | 10 mm grid of M2.5 holes |

A Jetson Orin Nano does **not** fit: it needs the whole rim tray. The Pi 5 is the planned computer.

## Found and fixed on the way

The first chest design was bolted through the torso shelf at 4 points. The shelf is an open X of ribs:
the front two bolts went through empty space, and the chest legs cut into the tray rim. Now the chest's
feet screw into heat-set inserts in the rack, and the rack is held by the 4 original computer bosses plus
4 M3 bolts through the torso's solid top plate. Strength-checked: SF ≥ 4.7.

## Cable instead of on-board?

| Option | On the robot | Verdict |
|---|---|---|
| All on board (Pi + battery) | everything | **The plan.** Free to walk and do chores; the model assumes it |
| Power over a cable, Pi on board | Pi; battery replaced by a 180 g dummy | **Best for bench testing.** The power PCB already has a 12 V DC input. Keep the dummy: the policy was trained with that mass |
| Computer and power over cables | servo adapter + IMU only | Possible, not recommended. The cable pulls the robot sideways (not in training), the runtime needs porting to a laptop, and Wi-Fi is too jittery for a 50 Hz loop (USB only) |

## Mass model

The rack (25 g), battery (180 g) and Pi (65 g) are modelled explicitly. The original torso model already
contained ~97 g of stand-in blocks for its electronics in the tray; that share is subtracted. The robot is
2.567 kg in the model. **Weigh your finished torso before training** and correct the model if it differs:
battery and computer sizes and masses are constants in `hardware/cad/parametric/chhaya_cad.py`
(`BATT`, `BATT_KG`, `PI`, `PI_KG`).
