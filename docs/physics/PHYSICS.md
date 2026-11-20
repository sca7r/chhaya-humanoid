# Physics checks: 20-joint model (legs + arms), MuJoCo

All 20 joints are driven exactly like `biped_env_v2` drives the legs: the BAM STS3215 model
(`actuator/params/sts3215_12v/m6.json`, stall capped at 2.94 N·m, firmware P 32, 1-count dead
zone, frictionloss/damping 0 on the servo dofs, armature from the fit), 200 Hz physics,
50 Hz targets. **There is no policy and no balance controller.** The legs just hold the
crouched home pose (hip pitch −10°, knee −26°, ankle pitch −16°), so these tests check the
model, not a controller. Reproduce with `python -m chhaya.physics_check` from the
`training/` folder. The raw numbers are in `physics_report.json`.

| Test | Result |
|---|---|
| T0 model sanity | ✅ compiles (MuJoCo 3.2.7 and 4.0); all inertias physical; every arm joint's sign matches the docs |
| T1 stand 6 s from a drop | ✅ settles at 0.279 m and 0.22° tilt; 0 mm drift; no servo saturates; no arm contacts. Peak leg torque at rest 0.37 N·m|
| T2 holding worst arm poses | ✅ T-pose / arms forward / arms up: arm servos at most 6.7 % of stall; tracking error ≤ 1.1°; robot tilt ≤ 0.7° |
| T3 arm tracking | ✅ 0.5 Hz, ±0.6–0.8 rad: 2.7–3.5° RMS, 40 ms lag, no saturation. At 1.5 Hz the command needs ~7.5 rad/s, beyond the servo's ~4.9 rad/s no-load speed, so it saturates. That is the real servo's limit, not a model fault |
| T4 arms moving while standing | ✅ arm swing 1.5 Hz, wave, raise/lower, fast random arms: never falls; peak tilt 1.3°; leg servos never saturate; no arm contacts |
| T5 leg clearance | ✅ arms at rest, 2000 random leg poses within gait ranges: **0 %** contact. Within the legs' full soft limits (e.g. 45° hip abduction): 12.5 % of poses touch, which is real and also happens on a person |
| T6 robustness | ✅ half timestep (2.5 ms): identical result; backlash hinges (1.5°) on all 20 joints: stable, 2.9° tilt |
| T7 torque off | ✅ no blow-up. The gearbox friction in the model holds the stance (as the real servos' gearboxes do) |
| T8 shove and fall | ✅ no NaN, contacts behave. Worst arm penetration into the floor is 9.9 mm. Peak arm joint torque on impact is 4.0 N·m (friction + motor), so falls, not stall, are the bracket design load: print in PETG |
| T9 speed | raw `mj_step` 18k/s (64 geoms, 26 dofs). Inside the env the Python BAM servo step dominates (~4.6k steps/s), so training throughput in MuJoCo is about unchanged |

**Verdict:** the model is physically sound and ready for policy work. The training-code side
(BAM on the arm joints, arm rest pose as home, scripted arm motion, mass randomisation, USD) is
implemented: see `docs/TRAINING.md`.

Run on the current design (rounded arms, claws with micro servos as fixed mass, halo head;
2.423 kg after the 2026-11-20 torso mass correction) in the Isaac Lab 3.0 environment (Python 3.12, numpy 2,
MuJoCo 3.12): nothing falls or goes unstable in any test.
