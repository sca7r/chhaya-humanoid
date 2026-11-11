# Upper body (arms + head) in the training code

The robot with the upper body has 20 joints: the 12 legs plus 8 arm joints (`{l,r}_shoulder_pitch`,
`{l,r}_shoulder_roll`, `{l,r}_arm_yaw`, `{l,r}_elbow`). The halo head is fixed. Each hand has a one-servo
claw, which is **not** a simulated joint: its micro servo and jaws are fixed mass on the forearm link, so
the training code never sees it. The robot weighs 2.39 kg (the legs-only robot is 1.66 kg). CAD, URDF and the design notes live in this repository under
`upper_body/` (`README.md` there).

## How it is trained: legs-only policy, scripted arms

**The policy is unchanged.** It sees and commands the same 12 leg joints in the same order
(`utils.residual_home.POLICY_JOINT_NAMES`). So:
- a legs-only checkpoint can be resumed on the robot with arms (same observation and action sizes);
- the exported bundle and the Jetson/Pi runtime don't change;
- the real robot can later run this policy with cheap scripted arm servos.

**The arms are a moving payload.** All 8 arm joints use the same BAM STS3215 servo model as the legs
(Isaac: a second `BamActuatorCfg` group `arms`; MuJoCo: a second `BamServo`). At every reset each env
draws one arm mode (`envs/isaaclab/mdp/arm_motion.py`):

| Mode | What the arms do | Default share |
|---|---|---|
| rest | rest pose (roll 0.12, elbow −0.35) plus a small per-episode offset | 30 % |
| swing | counter-swing synced to the gait clock, random amplitude | 30 % |
| random | smooth band-limited random motion inside safe ranges | 25 % |
| hold | random poses, re-drawn every 1–3 s, smoothly blended | 15 % |

Targets are rate-limited to 4 rad/s (what the servo can follow) and clamped to the joint limits.
With domain randomisation on, the arm link masses are scaled ×0.9–1.1. The policy therefore learns
to walk while the arms do anything the real robot's scripts might do.

## Turning it on

Everything is behind `env_config.upper_body` in the YAML. Without that block, nothing changes: the
legs-only robot and all existing configs behave exactly as before (the test suite confirms this).

```yaml
env_config:
  upper_body:
    enabled: true
    arm_motion: {mode_probs: {rest: 0.3, swing: 0.3, random: 0.25, hold: 0.15}, max_speed_rad_s: 4.0, ...}
    mass_scale_range: [0.9, 1.1]   # arm mass DR (training only)
    mujoco_arm_mode: rest          # what the MuJoCo eval/gate does with the arms: rest | swing | random
```

`config/config_upper_body.yaml` is `config.yaml` plus this block (and `model_name: Chhaya_SAC`).

```bash
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/config_upper_body.yaml --headless
```

## What changed (file by file)

| File | Change |
|---|---|
| `envs/assets/robotV2/robot_upper.usd` | the robot with arms. A thin layer over `robot.usd`: the legs are the exact prims the old policies trained on, the chest and head are merged into `base_link`, and the meshes are embedded (no external files). Built with plain usd-core by `hardware/upper_body/tools/build_usd.py` |
| `envs/assets/robotV2/biped_robot_upper.py` | `BIPED_ROBOT_UPPER_CFG`: the USD, arm rest pose in `init_state`, actuator groups `legs` (first) and `arms` |
| `envs/assets/robotV2/robot_v2_upper_sts3215.xml` | MuJoCo model with arms (meshes from `robot_hardware/robot_description/meshes/upper_sts3215/`) |
| `envs/isaaclab/mdp/arm_motion.py` | arm target generator (pure torch, unit-tested) |
| `envs/isaaclab/biped_env.py` | "policy joints" vs articulation joints: every joint-vector term (obs, actions, limits, defaults, rewards, imitation, resets) uses the 12 policy joints, and the arms are driven from `arm_motion`. With `policy_joint_names = None` (legs-only robot) every mapping is the identity |
| `envs/isaaclab/biped_env_cfg.py` | `policy_joint_names`, `arm_joint_names`, `arm_motion`, `arm_body_regex` (defaults: off) |
| `src/isaaclab/env_config.py` | `_apply_upper_body`: swaps in the robot with arms, sets the fields above and adds the `randomize_arm_mass` event. `_apply_action_home` keeps the arm init entries |
| `config/schema.py` | `env_config.upper_body` |
| `envs/mujoco/biped_env_v2.py` | with `upper_body.enabled`: loads the upper MJCF and runs the arm servos (BAM) toward rest / swing / random targets. Policy interface unchanged |
| `src/mujoco/policy_gate.py` | the arm links don't count as legs |
| `upper_body/` | joint tables (`model.py`), scripted motions (`motions.py`), renders, `physics_check.py` |
| `tests/test_arm_motion.py`, `tests/test_upper_body_mujoco.py` | 11 tests |

## Verified here (no GPU)

- MuJoCo, with the repo's pins (numpy 1.26.3, MuJoCo 3.2.7):
  - the physics checks in `physics_report/PHYSICS.md` all pass;
  - the 11 new tests pass;
  - the rest of the test suite gives exactly the same results as the unmodified upstream code
    (7 failures there are upstream's own gait-table and config expectations).
- USD: 21 rigid bodies, 20 revolute joints, 2.392 kg, joint frames matching the URDF to 0.0 mm.
- Isaac-side code: compiles and passes pyflakes. It was **not run**: Isaac Sim needs an RTX GPU.
  The first thing to do on the GPU machine is the smoke test in `GPU_RUNBOOK.md`.

## Known limits

- `src/isaaclab/check_actuator.py`, `jitter_probe.py` and `make_fall_bank.py` assume the legs-only
  robot (12 joints). Use them with `config.yaml`, not the upper-body config.
- The get-up task and the fall bank were not adapted to the arms.
- A whole-body policy (arms in the action space) is not implemented. The arms stay scripted, as
  planned for the real build.
- The MuJoCo eval keeps the arms at rest by default. Set `mujoco_arm_mode: swing` or `random` to
  test robustness in the gate.
