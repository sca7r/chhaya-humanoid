# Printing

All parts are in [`stl/`](stl/), one file per part, in millimetres.

### Legs, hips and torso

The legs are mirror images, so most files are printed twice; only the upper hips and the feet have
separate left and right files.

| File | Qty | Notes |
|---|---|---|
| `body.stl` | 1 | |
| `IMU_tray.stl` | 1 | |
| `ankle_and_hip.stl` | **4** | Lower hip and ankle are the same part. **Critical**, see below |
| `thigh.stl` | 2 | |
| `calf.stl` | 2 | |
| `left_upper_hip.stl` | 1 | |
| `right_upper_hip.stl` | 1 | |
| `left_foot.stl` | 1 | |
| `right_foot.stl` | 1 | |
| **Legs, hips and torso** | **14 parts** | About 0.5 kg of PLA |

### Chest, head, arms and claws

Files are named `l_...` / `r_...` for the left and right arm. They are saved in robot coordinates,
so lay each one down in the slicer as the table says: the orientation decides which way the layers
run, and the strength check ([`docs/design.md`](../docs/design.md)) assumes these.

| File(s) | Qty | Orientation and notes |
|---|---|---|
| chest_frame | 1 | standing on its feet. supports under the saddle and bottom plate |
| head | 1 | standing on its flange. supports under the hoop; M2.5 grid for sensors |
| head_visor | 1 | flat. print in black, or cut from 1.5 mm tinted acrylic |
| l/r_shoulder_bracket | 1+1 | roll-servo plate flat on the bed |
| l/r_upper_arm (fork) | 1+1 | on its side, one ear flat on the bed. tree supports under the top ear; layer lines then run along the ears, which carry the load in-plane |
| l/r_elbow_link | 1+1 | elbow-servo plate flat on the bed |
| l/r_forearm (fork + shell) | 1+1 | on its side, one ear flat on the bed. supports inside the shell |
| l/r_claw_base | 1+1 | on its back (the housing's rear face on the bed). layers then run along the housing walls and the fixed finger, which is what the strength check assumes |
| l/r_claw_jaw | 1+1 | flat. 100 % infill |
| l/r_roll_idler_cap, l/r_elbow_idler_cap | 4 | plate flat, stub up. 100 % infill (it's tiny) |
| electronics_rack | 1 | plate flat on the bed, supports under the deck. 4 walls, 40 % infill: the chest stands on it |
| **Chest, head, arms, claws and rack** | **21 parts** | About 0.22 kg; 4 walls, 30 % gyroid, PLA or (better for falls) PETG |

Before printing the forks and pads, check the horn screw circle (`horn_pcd`, 14 mm) against your
servo horns, and the claw pocket (`GR_L`, `GR_W`: 22.8 x 12.2 mm body) against your micro servo.
A test print of `l_elbow_link.stl` checks the servo face-hole pattern cheaply.

## Settings

The reference robot was printed in PLA on a stock Creality Ender-3 V2 with the slicer's default PLA profile.
Nothing about this design needs special tuning: use your printer's default PLA settings, and add supports
where your slicer asks for them.

## The one part that matters: `ankle_and_hip.stl`

The lower-hip/ankle bracket is the part most likely to break when the robot falls. For all 4 copies:

- **Print it tilted 45°** on the bed, so the layer lines don't line up with the load from a fall.
- **Use at least 6 walls** (perimeters).

Print a spare or two: it's the part you'll replace first.
