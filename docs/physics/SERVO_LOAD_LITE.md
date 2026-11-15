# Servo load check

chhaya_lite.yaml: 2.567 kg, legs-only rig 1.661 kg. Servos: legs rds3225_6v8, arms mg996r_6v, supply 6.8 V. Leg stall 2.45 N m (continuous guide 0.82), arm stall 1.22 N m.

| Joint (left leg) | Stand, both feet | One foot, Chhaya | % of stall | % of continuous | One foot, legs-only rig |
|---|---|---|---|---|---|
| l_hip_yaw | 1.13 | 0.02 | 1 % | 3 % | 0.01 |
| l_hip_roll_joint | 0.55 | 1.34 | 55 % | 164 % | 0.87 |
| l_hip_pitch_joint | 0.85 | 0.04 | 2 % | 5 % | 0.03 |
| l_knee_joint | 0.68 | 0.35 | 14 % | 43 % | 0.13 |
| l_ankle_roll_joint | 0.55 | 1.37 | 56 % | 168 % | 0.82 |
| l_ankle_pitch_joint | 0.67 | 0.18 | 7 % | 22 % | 0.32 |

| Knee bend | Knee torque, both legs | Knee torque, one leg | % of stall (one leg) |
|---|---|---|---|
| 40 deg | 0.26 | 0.57 | 23 % |
| 60 deg | 0.37 | 0.83 | 34 % |
| 90 deg | 0.53 | 1.17 | 48 % |
| 120 deg | 0.65 | 1.44 | 59 % |

| Arm pose | Joint | Torque [N m] | % of stall |
|---|---|---|---|
| arm forward | l_shoulder_pitch | 0.20 | 16 % |
| arm forward | l_shoulder_roll | 0.00 | 0 % |
| arm forward | l_arm_yaw | 0.00 | 0 % |
| arm forward | l_elbow | 0.03 | 2 % |
| arm sideways | l_shoulder_pitch | 0.01 | 1 % |
| arm sideways | l_shoulder_roll | 0.19 | 16 % |
| arm sideways | l_arm_yaw | 0.00 | 0 % |
| arm sideways | l_elbow | 0.00 | 0 % |
