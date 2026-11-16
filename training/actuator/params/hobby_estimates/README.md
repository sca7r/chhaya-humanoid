# Hobby-servo parameter files (ESTIMATES, not measurements)

Used by `config/chhaya_lite.yaml`. Unlike `sts3215_12v/`, these were **not identified on a test rig**.
They put datasheet numbers into the same servo model (DC motor + proportional controller, model `m1`:
constant + viscous friction only).

| File | Servo | Voltage | Stall torque used | No-load speed | Stall current |
|---|---|---|---|---|---|
| `rds3225_6v8.json` | DSServo RDS3225 (dual shaft, 342:1 metal gears, 3-pole brushed motor) | 6.8 V | 25 kgf·cm = 2.45 N·m | 0.17 s/60° = 6.2 rad/s | 2.9 A |
| `mg996r_6v.json` | TowerPro MG996R | 6.0 V | 11 kgf·cm = 1.08 N·m | 0.14 s/60° = 7.5 rad/s | 2.5 A |

Datasheets checked 2026-11-16:
- RDS3225: speed 0.17 s/60° at 6.8 V (0.19–0.20 at 5 V), dead band 3 µs, pulse 500–2500 µs, 50–330 Hz,
  60 g, 40 × 20 × 40 mm, 180° and 270° versions. **Torque differs by source:** 28 kgf·cm at 6.8 V
  (servodatabase.com) and 34.5 kgf·cm (seller listings). The model uses the 25 kgf·cm the servo is sold
  as, the lowest figure, because cheap servos often miss their rating.
- MG996R (TowerPro sheet): 9.4 kgf·cm / 0.17 s/60° at 4.8 V, 11 kgf·cm / 0.14 s/60° at 6 V, dead band
  5 µs, 4.8–7.2 V, 55 g, 40.7 × 19.7 × 42.9 mm.
- An earlier version of the RDS3225 file used 0.13 s/60°, which was wrong.

Derived: `kt = V / no-load speed`, `R = 0.97 * V * kt / stall torque`.

Guessed (no data): `kd_fw` (derivative gain), `armature` (rotor inertia at the output; the RDS3225's
342:1 gearing is close to the STS3215's 345:1, whose measured value is 0.023, so 0.02 is plausible), `friction_base` (about 5 % of stall),
`friction_viscous`, `command_delay` (10 ms: half a 50 Hz PWM frame). `max_velocity = 100` switches off
the STS3215 firmware's speed limiter, which hobby servos do not have. The controller gain, dead band
and supply voltage are set in the config, with wide randomisation ranges because they are unknown.

Cheap servos often miss their datasheet torque and speed. Before trusting a policy trained on these
files, measure two real servos (pot tapped for feedback) with the rig in `actuator/` and replace them.
