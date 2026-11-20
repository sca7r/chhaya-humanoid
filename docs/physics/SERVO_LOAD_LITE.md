# Servo load check

chhaya_lite.yaml: 2.423 kg. Servos: legs rds3225_6v8, arms mg996r_6v, supply 6.8 V. Leg stall 2.45 N m (continuous guide 0.82), arm stall 1.22 N m.

| Joint (left leg) | Stand, both feet | One foot, Chhaya | % of stall | % of continuous |
|---|---|---|---|---|
| l_hip_yaw | 0.04 | 0.09 | 4 % | 11 % |
| l_hip_roll_joint | 0.08 | 1.24 | 51 % | 152 % |
| l_hip_pitch_joint | 0.20 | 0.34 | 14 % | 41 % |
| l_knee_joint | 0.26 | 0.28 | 11 % | 34 % |
| l_ankle_roll_joint | 0.02 | 1.15 | 47 % | 140 % |
| l_ankle_pitch_joint | 0.44 | 1.21 | 49 % | 148 % |

| Knee bend | Knee torque, both legs | Knee torque, one leg | % of stall (one leg) |
|---|---|---|---|
| 40 deg | 0.24 | 0.53 | 22 % |
| 60 deg | 0.35 | 0.78 | 32 % |
| 90 deg | 0.49 | 1.10 | 45 % |
| 120 deg | 0.61 | 1.35 | 55 % |

| Arm pose | Joint | Torque [N m] | % of stall |
|---|---|---|---|
| arm forward | l_shoulder_pitch | 0.20 | 16 % |
| arm forward | l_shoulder_roll | 0.00 | 0 % |
| arm forward | l_arm_yaw | 0.00 | 0 % |
| arm forward | l_elbow | 0.03 | 2 % |
| arm sideways | l_shoulder_pitch | 0.01 | 1 % |
| arm sideways | l_shoulder_roll | 0.19 | 15 % |
| arm sideways | l_arm_yaw | 0.00 | 0 % |
| arm sideways | l_elbow | 0.00 | 0 % |
