# Chhaya Lite: the budget variant with hobby servos (started 2026-11-15)

Two training tracks now run side by side from the same code:

| | Chhaya | Chhaya Lite |
|---|---|---|
| Config | `training/config/chhaya.yaml` | `training/config/chhaya_lite.yaml` |
| Leg servos (12) | STS3215, 12 V | RDS3225 (25 kgf·cm), 6.8 V |
| Arm servos (8) | STS3215 | MG996R (11 kgf·cm) |
| Claws (2, scripted, not simulated) | STS3215 | MG90S |
| Joint feedback | built-in 12-bit encoder over the serial bus | a wire soldered to each servo's pot, read by an ADC |
| Servo model | measured on a test rig | **estimated from datasheets** |
| Servo cost (prices from Harsh's table) | not checked | 12 × ₹1,150–1,370 + 8 × ₹270–320 + 2 × ₹185–240 ≈ ₹16,300–19,500 |
| Checkpoints | `checkpoints/Chhaya_SAC*` | `checkpoints/ChhayaLite_SAC*` |

Train either one with the same command (`ISAACLAB3.md` for the install):

```bash
cd training
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/chhaya.yaml       --num_envs 1024
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/chhaya_lite.yaml  --num_envs 1024
```

On one GPU run them one after the other; two at once share the memory and each gets slower.

## What the Lite simulation changes

Only the 17 config values marked `LITE` in `chhaya_lite.yaml` (a test checks that nothing else differs):

| Changed | Chhaya | Lite | Basis |
|---|---|---|---|
| Leg servo torque / speed | 2.94 N·m, 4.9 rad/s | 2.45 N·m, 8.1 rad/s at 6.8 V | datasheet |
| Arm servo torque / speed | 2.94 N·m, 4.9 rad/s | 1.22 N·m at 6.8 V, 7.5 rad/s at 6 V | datasheet |
| Supply | 10.2–12.6 V | 6.0–6.8 V | 2S battery through a regulator |
| Controller | P only, full drive at about 10° of error | P + D, full drive at about 3° (2–4.8° randomised) | guess |
| Dead band | ±0.09° | ±0.18° to ±0.35° | datasheet (3–5 µs) |
| Gear play | 1.0–2.5° | 1.0–3.0° | guess |
| Joint angle noise | 0.09° | 0.57° | guess for a pot through an ADC |
| Joint speed noise | 0.25 rad/s | 0.6 rad/s | guess (speed is computed from the noisy angle) |
| Zero offset per joint | about 1° | about 2° | coarse horn spline, pot non-linearity |
| Friction, rotor inertia | measured | guessed, friction randomised 0.6–1.5× | none |

Files: `training/actuator/params/hobby_estimates/` (the two servo files and how each number was
derived). The servo model gained two optional fields, `kd_fw` (derivative gain) and `v_nominal`; the
STS3215 files do not use them, so the Chhaya track is numerically unchanged.

**Not changed yet: the body.** Lite still uses the STS3215 frame's geometry and masses. The servos weigh
nearly the same (RDS3225 about 60 g, MG996R 55 g, STS3215 55 g), so the mass error is about 60 g in
2.57 kg. The frame redesign comes after this check says the servos are worth it.

## First results (CPU, no policy yet)

| Check | Chhaya | Lite |
|---|---|---|
| Worst static leg load, standing on one foot | 1.34 N·m = 46 % of stall | 1.37 N·m = 56 % of stall |
| Arm held straight out | 0.20 N·m = 7 % of stall | 0.20 N·m = 16 % of stall |
| Standing still in MuJoCo, no policy: joint jitter (RMS speed) | 0.008 rad/s | 0.13 rad/s |
| Isaac Lab 3.0, 4 robots, zero action for 1.2 s | 0 falls | 1 fall |
| Isaac Lab 3.0 environment builds and steps (20 actions, 370 observations) | yes | yes |

Full load table: `physics/SERVO_LOAD_LITE.md`.

Reading:
- **Torque is enough.** The RDS3225 legs and MG996R arms carry the static loads with margin.
- **The risk is jitter, not strength.** A stiff controller plus gear play makes the joints hunt. With no
  gear play the Lite model stands as quietly as the STS3215 one; with 2° of play it shakes about 16 times
  more. Whether a policy can walk on that is what training will show, and how big the effect really
  is depends on the guessed numbers above.

## Before building Lite

1. **Measure two real servos** (one RDS3225, one MG996R): tap the pot, run the pendulum rig in
   `training/actuator/`, fit the model, replace the estimated files. Until then a Lite policy is a
   feasibility result, not something to put on a robot.
2. **Electronics that differ from Chhaya:**
   - 20 analog inputs for the pots (a microcontroller board or ADC chips, 12-bit or better).
   - PWM with 1 µs resolution or better. A PCA9685 at 50 Hz has 4.9 µs steps, about 0.66° on a 270°
     servo, which is too coarse.
   - A 6.8 V supply sized for stall currents (several amps per servo; check the datasheets).
3. **Frame:** new servo pockets in `hardware/cad/parametric/chhaya_cad.py` and new legs. The legs are
   then no longer the design that has already walked.
