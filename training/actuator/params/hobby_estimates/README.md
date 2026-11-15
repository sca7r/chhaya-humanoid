# Hobby-servo parameter files (ESTIMATES, not measurements)

Used by `config/chhaya_lite.yaml`. Unlike `sts3215_12v/`, these were **not identified on a test rig**.
They put datasheet numbers into the same servo model (DC motor + proportional controller, model `m1`:
constant + viscous friction only).

| File | Servo | Voltage | Stall torque | No-load speed | From |
|---|---|---|---|---|---|
| `rds3225_6v8.json` | RDS3225 (25 kg, dual shaft) | 6.8 V | 25 kgf·cm = 2.45 N·m | 0.13 s/60° = 8.1 rad/s | seller datasheet |
| `mg996r_6v.json` | MG996R | 6.0 V | 11 kgf·cm = 1.08 N·m | 0.14 s/60° = 7.5 rad/s | seller datasheet |

Derived: `kt = V / no-load speed`, `R = 0.97 * V * kt / stall torque`.

Guessed (no data): `armature` (rotor inertia at the output), `friction_base` (about 5 % of stall),
`friction_viscous`, `command_delay` (10 ms: half a 50 Hz PWM frame). `max_velocity = 100` switches off
the STS3215 firmware's speed limiter, which hobby servos do not have. The controller gain, dead band
and supply voltage are set in the config, with wide randomisation ranges because they are unknown.

Cheap servos often miss their datasheet torque and speed. Before trusting a policy trained on these
files, measure two real servos (pot tapped for feedback) with the rig in `actuator/` and replace them.
