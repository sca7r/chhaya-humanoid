# Chhaya Lite: the budget variant with hobby servos (started 2026-11-15, datasheets checked 2026-11-16)

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
| Leg servo torque / speed | 2.94 N·m, 4.9 rad/s | 2.45 N·m, 6.2 rad/s at 6.8 V | datasheet (torque: the lowest of three published figures) |
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

## How it behaves (CPU, no policy yet)

| Check | Chhaya | Lite |
|---|---|---|
| Worst static leg load, standing on one foot | 1.34 N·m = 46 % of stall | 1.32 N·m = 54 % of stall |
| Arm held straight out | 0.20 N·m = 7 % of stall | 0.20 N·m = 16 % of stall |
| Standing with zero action in MuJoCo, 10 s | stands, joint jitter 0.012 rad/s | falls after 4.2 s, jitter 0.16 rad/s |
| Isaac Lab 3.0, 4 robots, zero action for 1.2 s | all stand | none fell, one was sagging |
| Isaac Lab 3.0 environment builds and steps (20 actions, 370 observations) | yes | yes |

Full load table: `physics/SERVO_LOAD_LITE.md`.

**Torque is enough; gear play decides everything.** Standing with zero action, by gear play and
controller stiffness (`python -m chhaya.lite_stand_scan`; cell = time of fall / leg joint jitter in rad/s,
"-" = stood for 10 s):

| Gear play | Full drive at 4.8° error | at 3.0° (nominal) | at 2.0° |
|---|---|---|---|
| 0.0° | - / 0.004 | - / 0.004 | - / 0.004 |
| 0.5° | - / 0.008 | - / 0.007 | - / 0.008 |
| 1.0° | - / 0.039 | - / 0.070 | - / 0.095 |
| 2.0° (nominal) | - / 0.105 | 4.2 s / 0.156 | 4.3 s / 0.199 |
| 3.0° | 3.5 s / 0.183 | 2.3 s / 0.246 | 2.2 s / 0.310 |

- With 0.5° of play or less, Lite is as quiet as the STS3215 robot.
- From about 2° the joints hunt (stiff controller pushing through the slack) and the robot shakes itself
  over without a policy. A trained policy closes the loop and may cope; that is what training will show.
- No datasheet gives the gear play of either servo. It is the first thing to measure on a real one:
  hold the horn, power the servo, and read the free movement at the end of a 100 mm arm
  (1.75 mm of movement = 1°).

## Before building Lite

1. **Measure two real servos** (one RDS3225, one MG996R): tap the pot, run the pendulum rig in
   `training/actuator/`, fit the model, replace the estimated files. Until then a Lite policy is a
   feasibility result, not something to put on a robot.
2. **Electronics that differ from Chhaya:**
   - 20 analog inputs for the pots (a microcontroller board or ADC chips, 12-bit or better).
   - PWM with 1 µs resolution or better. A PCA9685 at 50 Hz has 4.9 µs steps, about 0.66° on a 270°
     servo, which is too coarse.
   - PWM at 200–330 Hz for the RDS3225 (it accepts 50–330 Hz); the model assumes a 10 ms command delay.
3. **Servo versions and travel.** Hip pitch and knee move through 240°, so the legs need the **270°**
   RDS3225, not the 180° one. The shoulder pitch joint has a 229° range and the MG996R turns about 180°:
   limit that joint in the redesign, or use a 270° servo there.
4. **Current.** Stall is 2.9 A per RDS3225 and 2.5 A per MG996R (datasheets). All 20 never stall together;
   size the 6.8 V supply for about 20 A peaks.
5. **Frame:** new servo pockets in `hardware/cad/parametric/chhaya_cad.py` and new legs. The legs are
   then no longer the design that has already walked.

## A neural network for the servo or the noise?

Possible, and a known method (an "actuator network" trained on logged command and position data, then
used in place of, or on top of, the physics model). It cannot be trained from datasheets: it needs
recordings from the real servos, the same recordings the measurement in step 1 produces. Order of work:
record, fit the physics model first (few parameters, needs little data), then add a small network for
whatever the physics model still gets wrong.

Datasheet sources: [RDS3225 at servodatabase.com](https://servodatabase.com/servo/dsservo/rds3225),
[RDS3225 seller listing](https://aifitlab.com/products/dsservo-rds3225-servo-motor),
[MG996R TowerPro sheet](https://people.ece.cornell.edu/land/courses/ece4760/FinalProjects/f2017/kmv48_rp625/kmv48_rp625/kmv48_rp625/MG996RTowerPro.pdf).
