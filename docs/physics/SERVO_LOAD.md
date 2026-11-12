# Servo load check

Chhaya 2.567 kg, legs-only rig 1.661 kg. STS3215 at 12 V: stall 2.94 N m, continuous guide 0.98 N m.

| Joint (left leg) | Stand, both feet | One foot, Chhaya | % of stall | % of continuous | One foot, legs-only rig |
|---|---|---|---|---|---|
| l_hip_yaw | 0.00 | 0.01 | 0 % | 1 % | 0.01 |
| l_hip_roll_joint | 0.01 | 1.34 | 46 % | 137 % | 0.87 |
| l_hip_pitch_joint | 0.07 | 0.12 | 4 % | 12 % | 0.03 |
| l_knee_joint | 0.04 | 0.15 | 5 % | 15 % | 0.13 |
| l_ankle_roll_joint | 0.04 | 1.26 | 43 % | 129 % | 0.82 |
| l_ankle_pitch_joint | 0.30 | 0.58 | 20 % | 59 % | 0.32 |

| Knee bend | Knee torque, both legs | Knee torque, one leg | % of stall (one leg) |
|---|---|---|---|
| 40 deg | 0.26 | 0.57 | 19 % |
| 60 deg | 0.37 | 0.83 | 28 % |
| 90 deg | 0.53 | 1.17 | 40 % |
| 120 deg | 0.65 | 1.44 | 49 % |

| Arm pose | Joint | Torque [N m] | % of stall |
|---|---|---|---|
| arm forward | l_shoulder_pitch | 0.20 | 7 % |
| arm forward | l_shoulder_roll | 0.00 | 0 % |
| arm forward | l_arm_yaw | 0.00 | 0 % |
| arm forward | l_elbow | 0.03 | 1 % |
| arm sideways | l_shoulder_pitch | 0.01 | 0 % |
| arm sideways | l_shoulder_roll | 0.19 | 7 % |
| arm sideways | l_arm_yaw | 0.00 | 0 % |
| arm sideways | l_elbow | 0.00 | 0 % |
