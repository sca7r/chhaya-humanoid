# 02 Servo setup

Do this **before** mounting any servo in the structure. A servo with the wrong ID or a horn on the wrong
spline is the most common cause of a robot that moves wrongly, and it is hard to fix once the leg is closed.

Set the IDs and centre the servos on a PC: plug the Waveshare bus adapter in by USB and use Feetech's FD debug
software (Windows). Calibration needs the assembled robot and the on-robot runtime (step 06).

## 1. Set the IDs

New STS3215s all ship as ID 1, so they cannot share the bus until each has its own ID.

1. Connect **one servo at a time** to the bus.
2. In FD, search the bus, select the servo and write its new ID from the map below.
3. Label the servo with its ID and joint (tape on the case).

| ID | Joint | ID | Joint |
|---|---|---|---|
| 13 | `l_hip_yaw` | 7 | `r_hip_yaw` |
| 2 | `l_hip_roll_joint` | 8 | `r_hip_roll_joint` |
| 3 | `l_hip_pitch_joint` | 9 | `r_hip_pitch_joint` |
| 4 | `l_knee_joint` | 10 | `r_knee_joint` |
| 5 | `l_ankle_roll_joint` | 11 | `r_ankle_roll_joint` |
| 6 | `l_ankle_pitch_joint` | 12 | `r_ankle_pitch_joint` |
| 14 | `l_shoulder_pitch` | 18 | `r_shoulder_pitch` |
| 15 | `l_shoulder_roll` | 19 | `r_shoulder_roll` |
| 16 | `l_arm_yaw` | 20 | `r_arm_yaw` |
| 17 | `l_elbow` | 21 | `r_elbow` |

## 2. Centre before mounting the horn

Before fitting a horn, centre the servo (position 2048, the middle of its 0-4095 range) so that the joint
has travel on both sides. The calibration in step 3 takes care of the exact offset.

## 3. Calibrate (after assembly)

Each joint needs two numbers: its zero (the servo position at the joint's URDF zero) and its direction sign.
The policy works relative to the standing pose, so an error here shifts every observation and command.
Calibrate with the robot hanging and the torques off, and again whenever you reseat a horn. The on-robot
runtime for Chhaya, which does this step, is not written yet (step 06).

## 4. PID

The PID is P-only (`p: 32, d: 0, i: 0`) on purpose: the simulated servo model the policy was trained with
is a P-only firmware loop. The runtime refuses to arm a policy whose required PID does not match the config.
