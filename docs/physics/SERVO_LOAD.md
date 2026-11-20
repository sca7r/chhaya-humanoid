# Servo load check

chhaya.yaml: 2.423 kg. Servos: legs sts3215_12v, arms sts3215_12v, supply 12 V. Leg stall 2.94 N m (continuous guide 0.98), arm stall 2.94 N m.

| Joint (left leg) | Stand, both feet | One foot, Chhaya | % of stall | % of continuous |
|---|---|---|---|---|
| l_hip_yaw | 0.01 | 0.01 | 0 % | 1 % |
| l_hip_roll_joint | 0.01 | 1.26 | 43 % | 128 % |
| l_hip_pitch_joint | 0.07 | 0.12 | 4 % | 12 % |
| l_knee_joint | 0.04 | 0.13 | 4 % | 13 % |
| l_ankle_roll_joint | 0.03 | 1.19 | 40 % | 121 % |
| l_ankle_pitch_joint | 0.28 | 0.55 | 19 % | 57 % |

| Knee bend | Knee torque, both legs | Knee torque, one leg | % of stall (one leg) |
|---|---|---|---|
| 40 deg | 0.24 | 0.53 | 18 % |
| 60 deg | 0.35 | 0.78 | 26 % |
| 90 deg | 0.49 | 1.10 | 37 % |
| 120 deg | 0.61 | 1.35 | 46 % |

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
