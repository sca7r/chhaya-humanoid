# 05 Electronics

Reference: [`electronics/wiring/`](../../electronics/wiring/) (pinouts, servo IDs) and
[`electronics/power_pcb/`](../../electronics/power_pcb/) (the board).

## 1. Power PCB

Order it or build it (see the [PCB page](../../electronics/power_pcb/README.md#getting-one)), and fit a
blade fuse in each holder.

- **BATTERY** (XT60): the 3S LiPo.
- **SWITCH**: your power switch. It is in series with both outputs, so it turns the whole robot on and off.
- **SERVOS**: servo bus power.
- **JETSON** (the computer output): the 12 V to 5 V step-down. From its 5 V terminals: the USB-C pigtail to the
  Raspberry Pi 5, and the two claw servos' power.

Before connecting anything, check the polarity of every output with a multimeter.

## 2. Servo bus

Daisy-chain the servos of each limb with their cables (four chains: two legs, two arms) and bring each chain to
the bus adapter on the rack with a 30 cm extension lead.

## 3. Raspberry Pi and servo adapter

Waveshare Bus Servo Adapter (A), wired **straight through**: Pi pin 8 (TX) to adapter TX, pin 10 (RX)
to adapter RX, pin 6 to GND. Claw servos: signal to pins 32 (left) and 33 (right), power from the step-down's 5 V terminals
(see [wiring](../../electronics/wiring/README.md#raspberry-pi-5-header)).

## 4. IMU

BNO055 on I2C bus 1 (pin 3 SDA, pin 5 SCL), address 0x28, in `IMU_tray`. Mount it rotated 180° about the
robot's X (forward) axis, as on the reference build. The runtime corrects for exactly that mounting; see
[wiring](../../electronics/wiring/README.md#imu-mounting) if yours differs.

## Check

Power on with the Pi disconnected first and measure the SERVOS and JETSON outputs: both should read the battery
voltage (11.1-12.6 V for a 3S LiPo). Then check the step-down's output reads 5.0-5.2 V before plugging in the Pi.
