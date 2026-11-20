# 07 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No servo answers at any baud rate | UART TX/RX crossed | The Waveshare adapter is straight-through: TX to TX, RX to RX |
| Port opens but nothing is sent | Header UART not enabled | Add `dtparam=uart0=on` to `/boot/firmware/config.txt`, reboot; `/dev/ttyAMA0` must exist |
| Two servos fight / bus errors | Two servos share an ID (often a new one still at ID 1) | Reassign IDs one servo at a time |
| A joint moves the wrong way | `direction_sign` wrong | Flip it during calibration ([02](02_servo_setup.md#3-calibrate-after-assembly)) |
| Robot leans or the policy behaves oddly | `default_position_deg` wrong for some joint, or a horn reseated without recalibrating | Recalibrate (step 02) |
| IMU tilt has the wrong sign | IMU mounted differently from the reference build | See [IMU mounting](../../electronics/wiring/README.md#imu-mounting) |
| BNO055 not at 0x28 | ADDR pin high, or wiring | Check `i2cdetect -y 1`; ADDR low gives 0x28 |
