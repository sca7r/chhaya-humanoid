# Wiring

A full wiring diagram is coming soon. Until then, this page has everything needed to wire the robot.

## Overview

```
3S LiPo (XT60) ──► Power PCB ──┬──► SERVOS terminal ──► servo bus power (20× ST3215: 4 chains, one per limb)
                    (fused,    └──► computer terminal (labelled JETSON) ──► 12 V to 5 V step-down ──► Pi 5 USB-C
                     switched)
Pi UART0 (pins 8/10) ──► Waveshare Bus Servo Adapter (A) ──► servo bus data line
Pi I2C1 (pins 3/5)   ──► BNO055 IMU (address 0x28)
Pi PWM (pins 32/33)  ──► left / right claw micro servo (5 V from the Pi's 5 V pins)
```

The legs-only reference robot used a Jetson Orin Nano with the same pin numbers; it does not fit on Chhaya's
electronics rack.

## Servo ID map

All 20 servos share one half-duplex bus at 1 Mbaud. IDs are set once per servo (see
[build guide step 2](../../docs/build/02_servo_setup.md)). The leg values are the ones in
BipedRobotJetson's `config/robot.yaml`; set the arm signs during calibration (positive = the joint's positive
direction in `robot_description`).

| ID | Joint | Limb | direction_sign |
|---|---|---|---|
| 13 | `l_hip_yaw` | Left leg | +1 |
| 2 | `l_hip_roll_joint` | Left leg | −1 |
| 3 | `l_hip_pitch_joint` | Left leg | −1 |
| 4 | `l_knee_joint` | Left leg | −1 |
| 5 | `l_ankle_roll_joint` | Left leg | −1 |
| 6 | `l_ankle_pitch_joint` | Left leg | −1 |
| 7 | `r_hip_yaw` | Right leg | −1 |
| 8 | `r_hip_roll_joint` | Right leg | +1 |
| 9 | `r_hip_pitch_joint` | Right leg | +1 |
| 10 | `r_knee_joint` | Right leg | +1 |
| 11 | `r_ankle_roll_joint` | Right leg | +1 |
| 12 | `r_ankle_pitch_joint` | Right leg | +1 |
| 14 | `l_shoulder_pitch` | Left arm | at calibration |
| 15 | `l_shoulder_roll` | Left arm | at calibration |
| 16 | `l_arm_yaw` | Left arm | at calibration |
| 17 | `l_elbow` | Left arm | at calibration |
| 18 | `r_shoulder_pitch` | Right arm | at calibration |
| 19 | `r_shoulder_roll` | Right arm | at calibration |
| 20 | `r_arm_yaw` | Right arm | at calibration |
| 21 | `r_elbow` | Right arm | at calibration |

No joint uses ID 1, the factory default, so a servo that still answers on ID 1 is one you have not set yet.
Each limb is one daisy chain; the arm chains start at the shoulder-pitch servos in the chest, the leg chains at
the hip-yaw servos in the torso. The rack is a few centimetres from each chain's first servo, so 30 cm leads
reach the adapter.

## Raspberry Pi 5 header

| Pin | Signal | Goes to |
|---|---|---|
| 8 | UART0 TX (`/dev/ttyAMA0`, enable with `dtparam=uart0=on`) | Waveshare adapter **TX** |
| 10 | UART0 RX | Waveshare adapter **RX** |
| 6 | GND | Waveshare adapter GND |
| 3 | I2C1 SDA | BNO055 SDA |
| 5 | I2C1 SCL | BNO055 SCL |
| 1 | 3.3 V | BNO055 VIN |
| 9 | GND | BNO055 GND |
| 32 | GPIO12, hardware PWM | left claw servo signal |
| 33 | GPIO13, hardware PWM | right claw servo signal |
| 2, 4 | 5 V | claw servos' + (both) |
| 34, 39 | GND | claw servos' − |

The Waveshare adapter is wired **straight through** (TX to TX, RX to RX), not crossed like most UARTs.
Crossing them gives complete silence on the bus.

The Pi's 5 V pins are its input rail, so the claw servos draw from the 5 A step-down directly (an MG90S stalls at
about 0.7 A; the Pi itself needs about 2 A here). Their 3.3 V PWM signal drives an MG90S reliably.

## IMU mounting

The BNO055 is mounted rotated **180° about the body X axis** (forward) relative to the robot frame
(X forward, Y left, Z up). The runtime corrects for it by negating Y and Z of the accelerometer and
gyro. If you mount it the other way up, change `_MOUNT_FLIP_YZ` in BipedRobotJetson's
`hardware/imu/bno055.py` to `+1.0` and re-check with the self-test.

## Power PCB connectors

| Connector | Type | Goes to |
|---|---|---|
| BATTERY | XT60 | 3S LiPo |
| WALL_DC | DC barrel jack | 12 V wall adapter (optional, for bench work) |
| SWITCH | 2-pin 5.0 mm screw terminal | Power switch, in series: turns both outputs on and off |
| SERVOS | 2-pin 5.0 mm screw terminal | Servo bus power |
| JETSON | 2-pin 5.0 mm screw terminal | Computer power: the 12 V to 5 V step-down for the Pi 5 |

Details in [`../power_pcb/`](../power_pcb/).
