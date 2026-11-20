# 06 Software and first power-on

## 1. Set up the Raspberry Pi

Flash Raspberry Pi OS (64-bit, Lite is enough) with Raspberry Pi Imager. Then:

- enable the header UART: add `dtparam=uart0=on` to `/boot/firmware/config.txt` and reboot. It appears as
  `/dev/ttyAMA0` (pins 8 and 10);
- enable I2C (`sudo raspi-config` → Interface Options → I2C);
- add your user to `dialout` and `i2c`, and check the IMU with `i2cdetect -y 1` (expect `28`).

## 2. The runtime

Chhaya's on-robot runtime is **not written yet**. It will read the exported policy bundle (`policy.onnx` +
`deploy_config.json`), talk to the 20 servos over the bus and the BNO055 over I2C, drive the claws, and run
the policy at 50 Hz. This page will cover installing it, calibration ([02](02_servo_setup.md#3-calibrate-after-assembly)),
and the first power-on.

## 3. First power-on, when the runtime exists

1. Hang the robot (or hold it) with the feet off the ground for every first test.
2. Check that all 20 servos report position, voltage and temperature, and calibrate.
3. Torque on, move to the standing pose, check every joint goes where the model says.
4. Lower it to the ground with a hand on it and let it stand. Only then run a policy.
