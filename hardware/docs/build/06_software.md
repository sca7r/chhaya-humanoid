# 06 Software and first power-on

The on-robot runtime is [BipedRobotJetson](https://github.com/AsterisCrack/BipedRobotJetson); its README has the
full details. It was written for a Jetson and runs on a Raspberry Pi 5 with two settings changed (below). This
page is the order to do things in on a new robot.

## 1. Set up the Raspberry Pi

Flash Raspberry Pi OS (64-bit, Lite is enough) with Raspberry Pi Imager. Then:

- enable the header UART: add `dtparam=uart0=on` to `/boot/firmware/config.txt` and reboot. It appears as
  `/dev/ttyAMA0` (pins 8 and 10);
- enable I2C (`sudo raspi-config` → Interface Options → I2C);
- add your user to `dialout` and `i2c`, and check the IMU with `i2cdetect -y 1` (expect `28`).

## 2. Install the runtime

```bash
git clone --recurse-submodules https://github.com/AsterisCrack/BipedRobotJetson
cd BipedRobotJetson
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-hardware.txt   # remove Jetson.GPIO from this file first
```

Set `UART_PORT=/dev/ttyAMA0`, and `i2c_bus: 1` in `config/hardware.yaml`.

## 3. First power-on, robot hanging

Hang the robot (or hold it) with the feet off the ground for every first test.

1. Start the server and open the web UI. All 12 leg servos should report position, voltage and temperature.
2. Calibrate (see [02 Servo setup, step 3](02_servo_setup.md#3-calibrate-after-assembly)).
3. Run the self-test (torque off).

## 4. Safe standing test

1. Robot hanging, torque on, Home → `stand`. Check every joint goes where the 3D view says.
2. Lower it to the ground with a hand on it, and let it stand.
3. Only then deploy a policy (BipedRobotJetson README, "Deploying a policy").

The runtime drives the 12 leg servos. Chhaya's 20-joint policy also needs the 8 arm servos and the claws
added to it, which is not written yet ([04](04_chest_arms_head.md#software)). The Raspberry Pi setup has not
been tested on a robot yet.
