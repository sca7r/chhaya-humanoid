# CAD

| File | |
|---|---|
| [`step/robot_assembly.zip`](step/robot_assembly.zip) | The legs, hips and torso assembly as STEP (zipped: the raw file is 160 MB). The chest, head and arms are not in it: see `robot_description/urdf/chhaya.urdf` for the whole robot |
| [`step/bodies/`](step/bodies/) | One STEP per printed part |
| [`parametric/chhaya_cad.py`](parametric/chhaya_cad.py) | CadQuery source of the chest, head, arms and claws: every dimension is a named constant. Their STEP files in `step/bodies/` are generated from it (`../regenerate.sh`) |
| [`parametric/checks/`](parametric/checks/) | Results of the automatic checks on those parts: overlap, clearance, strength |

STEP opens in any CAD tool (FreeCAD, Onshape, SolidWorks, Fusion 360...). Units are millimetres, and the
robot frame follows the URDF: X forward, Y left, Z up.

## Parts

| STEP | Printed as | Qty |
|---|---|---|
| `body.step` | `body.stl` | 1 |
| `IMU_tray.step` | `IMU_tray.stl` | 1 |
| `hip_and_ankle.step` | `ankle_and_hip.stl` | 4 |
| `thigh.step` | `thigh.stl` | 2 |
| `calf.step` | `calf.stl` | 2 |
| `left_upper_hip.step`, `right_upper_hip.step` | `left_upper_hip.stl`, `right_upper_hip.stl` | 1 each |
| `left_foot.step`, `right_foot.step` | `left_foot.stl`, `right_foot.stl` | 1 each |

| `chest_frame.step`, `head.step`, `head_visor.step` | same names `.stl` | 1 each |
| `l_*.step`, `r_*.step` (shoulder_bracket, upper_arm, elbow_link, forearm, claw_base, claw_jaw, roll_idler_cap, elbow_idler_cap) | same names `.stl` | 1 each |

The legs are mirror images, and most parts are the same on both sides: print them twice. Only the upper hips
and the feet have separate left and right versions.
