# How Chhaya is trained

Chhaya has 20 servo joints: 12 in the legs and 8 in the arms (`{l,r}_shoulder_pitch`,
`{l,r}_shoulder_roll`, `{l,r}_arm_yaw`, `{l,r}_elbow`). The halo head is fixed. Each claw has a
micro servo that is **not** a simulated joint: it and the jaws are fixed mass on the forearm, so
the training code never sees it. The robot weighs 2.57 kg with its battery and Raspberry Pi 5. The mechanical design is in
`hardware/docs/design.md`.

## One whole-body policy for all 20 joints

Decided 2026-11-12: the policy learns the whole robot. The arms are no longer scripted.

**Action (20):** `[12 legs | 8 arms]`, each in [-1, 1]. Leg order is
`utils.residual_home.POLICY_JOINT_NAMES`; arm order is `ARM_JOINT_NAMES` in
`envs/assets/robotV2/chhaya_robot.py` (L,R interleaved per type, like the legs).
- legs: the unchanged residual map around the crouched home pose;
- arms: `target = clip(rest + arms.action_scale_rad[type] * a, joint limits)`, around the arm rest pose
  (roll 0.12, elbow −0.35), with the same action delay and EMA filter as the legs.

**Observation frame (74):** the 50-dim leg frame, unchanged, then arm `joint_pos − rest` (8),
arm `joint_vel` (8) and the previous arm actions (8). With 5 frames of history the actor sees 370
numbers. The critic sees the 74-dim frame plus the 10 privileged values (84).

**Servo model:** all 20 joints use the BAM STS3215 model (Isaac: actuator groups `legs` and `arms`;
MuJoCo: two `BamServo`s). Arm link masses are randomised ×0.9–1.1 in training.

**Rewards for the arms:** the arms have no task of their own yet. The walking rewards decide what
they do (they are free to counter-swing or catch balance), and four regularisers keep them sane:
`arm_action_rate_l2`, `arm_pos_l2` (pull to the rest pose), `arm_torques_l2`, `arm_pos_limits`.
The weights in `chhaya.yaml` are first guesses and have not been tuned on a GPU.

**Symmetry:** the mirror swaps left and right arm joints with sign +1 (the arm axes are mirrored in
the model; `tests/test_chhaya_mujoco.py::test_arm_mirror_is_a_plain_left_right_swap` checks it), so
the equivariant actor and the symmetry augmentation cover all 20 joints.

**What this policy does not do:** it walks with all 20 joints. It does not reach, grasp or do
chores. Those need their own tasks (commands for hand targets, rewards, objects in the scene),
which are not written. The claws are still fixed mass in simulation.

**A legs-only checkpoint cannot be resumed** into this policy: the network input and output sizes
differ (370 → 20 instead of 250 → 12).

## The old mode: legs-only policy, scripted arms

`arms.control: scripted` keeps the 12-joint policy and drives the arms from
`envs/isaaclab/mdp/arm_motion.py` as a moving payload (rest 30 %, gait-synced swing 30 %, random
25 %, hold 15 %, drawn per episode). It is still tested and is the fallback if the whole-body run
does not converge.

## Turning it on

```yaml
env_config:
  arms:
    enabled: true
    control: policy                # policy = whole-body (20 actions) | scripted = legs-only policy
    action_scale_rad: {shoulder_pitch: 1.0, shoulder_roll: 0.6, arm_yaw: 0.8, elbow: 1.0}
    mass_scale_range: [0.9, 1.1]   # arm mass DR (training only)
  reward_weights:
    arm_action_rate_l2: -0.01
    arm_pos_l2: -0.05
    arm_torques_l2: -2.0e-3
    arm_pos_limits: -1.0
```

A config without the `arms` block runs the legs-only rig (Chhaya without chest, head and arms),
which the regression tests use and which behaves exactly like the original Sir Servington code.
`config/chhaya.yaml` is Chhaya's config.

```bash
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/chhaya.yaml --headless
```

## Settings and hyperparameters (`config/chhaya.yaml`)

| Setting | Value |
|---|---|
| Algorithm | SAC (off-policy), entropy temperature auto-tuned |
| Actor / critic networks | MLP 1024-512-256 each; actor is exactly mirror-equivariant |
| Learning rates | actor 3e-4, critic 3e-4, no scheduler |
| Discount / n-step / target update | 0.99 / 5-step returns / τ = 0.01 |
| Replay buffer / batch | 1,000,000 transitions / 2048, 10 gradient batches every 4 env steps |
| Warm-up | 4096 random steps before learning |
| Smoothness | CAPS temporal 1.0, spatial 1.0 (σ 0.1) |
| Observation | 5-frame history, running normalisation, symmetry augmentation for the critic |
| Parallel robots | 4096 (Isaac Lab default for this task; `--num_envs` to change) |
| Control / physics rate | 50 Hz policy, 200 Hz physics (decimation 4), 20 s episodes |
| Terrain | rough terrain as randomisation, no terrain curriculum |
| Domain randomisation | servo model (voltage, stiffness, friction), masses, action delay, sensor noise, bias and lag, pushes |
| Checkpoints | `save_steps: 3000`, to `checkpoints/Chhaya_SAC*/` |

All of these except the arm block come unchanged from the legs-only setup, where the original
author tuned them against the real legs-only robot. Leave them alone for the first run.

## Where the arm support lives (paths under `training/`)

| File | Change |
|---|---|
| `envs/assets/robotV2/chhaya.usd` | Chhaya for Isaac Lab. A thin layer over `robot.usd`: the legs are the exact prims the old policies trained on, the chest and head are merged into `base_link`, and the meshes are embedded (no external files). Built with plain usd-core by `hardware/upper_body/tools/build_usd.py` |
| `envs/assets/robotV2/chhaya_robot.py` | `CHHAYA_ROBOT_CFG`: the USD, arm rest pose in `init_state`, actuator groups `legs` (first) and `arms` |
| `envs/assets/robotV2/chhaya.xml` | Chhaya for MuJoCo (meshes from `robot_hardware/robot_description/meshes/`) |
| `envs/isaaclab/mdp/arm_motion.py` | arm target generator (pure torch, unit-tested) |
| `envs/isaaclab/biped_env.py` | "policy joints" vs articulation joints: every leg joint-vector term (obs, actions, limits, defaults, rewards, imitation, resets) uses the 12 leg joints. Whole-body policy: the arm block is appended to the action and the frame (`_setup_arm_policy`, `_arm_obs`, `_arm_reward_terms`, `arm_actions_to_targets`, mirror tables). Scripted mode: the arms are driven from `arm_motion`. With `policy_joint_names = None` (legs-only robot) every mapping is the identity |
| `envs/isaaclab/biped_env_cfg.py` | `policy_joint_names`, `arm_joint_names`, `arm_motion`, `arm_body_regex` (defaults: off) |
| `src/isaaclab/env_config.py` | `_apply_arms`: swaps in the robot with arms, sets the fields above and adds the `randomize_arm_mass` event. `_apply_action_home` keeps the arm init entries |
| `config/schema.py` | `env_config.arms` |
| `envs/mujoco/biped_env_v2.py` | with `arms.enabled`: loads `chhaya.xml` and runs the arm servos (BAM). `control: policy`: 20 actions and the 74-dim frame, as in Isaac. `control: scripted`: rest / swing / random targets |
| `src/mujoco/policy_gate.py` | the arm links don't count as legs |
| `chhaya/` | joint tables (`model.py`), scripted motions (`motions.py`), renders, `physics_check.py` |
| `tests/test_arm_motion.py`, `tests/test_chhaya_mujoco.py`, `tests/test_whole_body_isaac_logic.py` | 19 tests |

## Verified here (no GPU)

- MuJoCo, with the repo's pins (numpy 1.26.3, MuJoCo 3.2.7):
  - the physics checks in `docs/physics/PHYSICS.md` all pass;
  - the 19 Chhaya tests pass (MuJoCo whole-body env; the Isaac env's arm action map and mirror tables, run on the CPU with the real methods lifted out of `biped_env.py`);
  - the rest of the test suite gives exactly the same results as the unmodified upstream code
    (7 failures there are upstream's own gait-table and config expectations).
- USD: 21 rigid bodies, 20 revolute joints, 2.567 kg, joint frames matching the URDF to 0.0 mm.
- Isaac-side code: compiles and passes pyflakes. It was **not run**: Isaac Sim needs an RTX GPU.
  The first thing to do on the GPU machine is the smoke test in `GPU_RUNBOOK.md`.

## Known limits

- `src/isaaclab/check_actuator.py`, `jitter_probe.py` and `make_fall_bank.py` assume the legs-only
  robot (12 joints). Use them with `config.yaml` (the legs-only rig), not `chhaya.yaml`.
- The get-up task and the fall bank were not adapted to the arms.
- **Not adapted to the whole-body policy yet:** `src/isaaclab/export_onnx.py` (it insists on a
  50-dim frame and 12 actions), `src/mujoco/policy_gate.py` (same), and the on-robot runtime (reads
  and commands 12 servos). Do these after the first training run shows the policy learns.
- With `control: scripted`: the MuJoCo eval keeps the arms at rest by default. Set `mujoco_arm_mode: swing` or `random` to
  test robustness in the gate.
