# Scripts

Every runnable script under `src/`, grouped by what you are doing:

| Workflow | Scripts |
|---|---|
| [Train](#train) | [`isaaclab/train.py`](#isaaclabtrainpy) |
| [Evaluate](#evaluate) | [`isaaclab/play.py`](#isaaclabplaypy), [`mujoco/sim2real.py`](#mujocosim2realpy) |
| [Export and deploy](#export-and-deploy) | [`isaaclab/export_onnx.py`](#isaaclabexport_onnxpy), [`isaaclab/verify_onnx_bundle.py`](#isaaclabverify_onnx_bundlepy), [`mujoco/policy_gate.py`](#mujocopolicy_gatepy) |
| [Get-up policy](#get-up-policy) | [`isaaclab/make_fall_bank.py`](#get-up-policy), `train.py` / `play.py` / `export_onnx.py` with `--task ChhayaGetUp`, [`mujoco/getup_gate.py`](#get-up-policy) |
| [Diagnose](#diagnose) | [`isaaclab/check_actuator.py`](#isaaclabcheck_actuatorpy), [`isaaclab/jitter_probe.py`](#isaaclabjitter_probepy), [`isaaclab/profile_train.py`](#isaaclabprofile_trainpy) |

The typical loop is **train → play → export → verify → gate**, with the diagnostics when something
looks wrong. See [`config/README.md`](../config/README.md) for the config file these scripts read.

## Conventions

- Run everything from `training/`.
- **Isaac Lab scripts** (train, play, check_actuator, jitter_probe, profile_train, the Isaac tools)
  need Isaac Lab's Python. They also accept Isaac Lab's own launcher flags such as `--headless`,
  `--device` and `--enable_cameras` (see `--help`).
- **The export, verify, gate and MuJoCo scripts** run without Isaac Lab.
- `--task` is `Chhaya` (walking, the default) or `ChhayaGetUp` (fall recovery).
- A checkpoint is `checkpoints/<run>/step_<N>.pt`. Next to it sit `config.yaml` (the config the run
  used, with `auto` values frozen) and `step_<N>_obs_scalers.pt` (observation statistics). The scripts
  below find these by themselves.
- Modules that are not scripts: [`isaaclab/env_config.py`](isaaclab/env_config.py) (turns
  `train.env_config` into the environment config, shared by every Isaac Lab script) and
  [`isaaclab/common.py`](isaaclab/common.py) (the environment wrapper used for training and playback).

---

## Train

### `isaaclab/train.py`

Train a policy in Isaac Lab. The algorithm, network, rewards and environment come from the config.

```bash
python src/isaaclab/train.py --task Chhaya --config_path config/chhaya.yaml --headless
```

| Argument | Default | Description |
|---|---|---|
| `--task` | `Chhaya` | `Chhaya` or `ChhayaGetUp` |
| `--config_path` | `config/chhaya.yaml` | Training config. With `--checkpoint` and no `--config_path`, the config saved in the checkpoint's folder is used |
| `--num_envs` | `4096` (env default) | Parallel robots |
| `--seed` | from config | Overrides `train.seed` |
| `--checkpoint` | none | Resume from a checkpoint: network weights and observation statistics are restored; the optimiser and replay buffer start fresh, and the step counter restarts at 0 |
| `--video` | off | Record clips during training into `runs/<run>/videos/` (turns cameras on) |
| `--video_length` | `200` | Frames per clip |
| `--video_interval` | `2000` | Steps between clips |

Outputs: TensorBoard logs in `runs/<model_name>_<timestamp>/`, checkpoints in
`checkpoints/<model_name>_<timestamp>/`, and the run's `config.yaml` beside them.

```bash
# Resume (the run's own config.yaml is picked up automatically)
python src/isaaclab/train.py --task Chhaya --checkpoint checkpoints/<run>/step_<N>.pt --headless

# Quick smoke test with few robots and a gait-table config
python src/isaaclab/train.py --task Chhaya --config_path config/chhaya.yaml --num_envs 64
```

---

## Evaluate

### `isaaclab/play.py`

Watch a checkpoint in Isaac Lab. It runs the nominal robot (no pushes, no physics randomization,
no action noise) through the same observation and actuation pipeline as training.

```bash
python src/isaaclab/play.py --task Chhaya --checkpoint checkpoints/<run>/step_<N>.pt
python src/isaaclab/play.py --task Chhaya --checkpoint ... --pushes --num_envs 4
python src/isaaclab/play.py --task Chhaya --checkpoint ... --teleop
```

| Argument | Default | Description |
|---|---|---|
| `--task` | `Chhaya` | `Chhaya` or `ChhayaGetUp` |
| `--checkpoint` | newest | Checkpoint to load. If omitted, the newest `.pt` in the newest folder under `checkpoints/` |
| `--config_path` | run's `config.yaml` | Otherwise `config/chhaya.yaml` |
| `--num_envs` | `1` | Robots to show |
| `--pushes` | off | Keep the training pushes on to watch push recovery |
| `--teleop` | off | Drive the command from the keyboard or a gamepad ([controls](#manual-control-and-pushes)). The camera follows env 0 and episodes don't time out |
| `--push_scale` | `1.0` | Size of the Space / pad A push, as a multiple of the training push range edge |
| `--seed`, `--video`, `--video_length`, `--video_interval` | | Accepted for symmetry with `train.py`. `--seed` only seeds the Space push, and `--video` only switches cameras and rendering on; `play.py` does not write clips |

Commands are sampled from the config's ranges unless `--teleop` is on. For a fixed command use `sim2real.py`.

Green is the command and blue is the measured base velocity. Both use one scale: an arrow for (vx, vy),
0.4 m long at the range's top speed, and an arc around the base for wz (270 deg at the top |wz|,
counter-clockwise for +wz). A green dot means the command has no translation. `sim2real.py` draws the same.

### `mujoco/sim2real.py`

Run an Isaac Lab policy in MuJoCo, a different simulator with its own servo implementation. This is
a sim-to-sim check, and it needs no Isaac Lab. It supports the V2 robot only.

```bash
python src/mujoco/sim2real.py --checkpoint checkpoints/<run>/step_<N>.pt --vx 0.1
python src/mujoco/sim2real.py --checkpoint ... --headless --num_episodes 5
python src/mujoco/sim2real.py --checkpoint ... --teleop
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | newest | Checkpoint to load |
| `--config_path` | run's `config.yaml` | Otherwise `config/chhaya.yaml` |
| `--task` | `Chhaya` | Robot |
| `--seed` | `42` | Random seed |
| `--headless` | off | No viewer window |
| `--num_episodes` | `0` | Episodes before exiting (0 = run until closed) |
| `--render_fps` | `50` | Target frame rate with a viewer |
| `--vx`, `--vy`, `--wz` | random | Fixed command per axis (m/s, m/s, rad/s) |
| `--teleop` | off | Drive the command from the keyboard or a gamepad ([controls](#manual-control-and-pushes)); needs the viewer, overrides `--vx/--vy/--wz`, and the camera tracks the robot |
| `--push_scale` | `1.0` | As in `play.py` |
| `--legacy-imu` | off | Old Isaac accelerometer convention; only for checkpoints trained before the step-averaged IMU fix |

### Manual control and pushes

`play.py` and `sim2real.py` share these bindings (`utils/teleop.py`), the same as the robot's web UI.
Keys need the sim window focused.

| Keyboard | Gamepad | Action |
|---|---|---|
| W / S | left stick up / down | forward / back |
| A / D | left stick left / right | left / right |
| Q / E | right stick left / right | turn left / right |
| Z / X | LB / RB | speed level -10 % / +10 % (10-100 %, starts at 50 %) |
| Space | A | random push (works without `--teleop` too) |

A held key commands speed level x the training config's range edge on that side. On the gamepad,
the level caps full stick deflection. The command is then limited exactly as in training
(`limit_shape`). The level and command are shown top-left in MuJoCo, and in a "Teleop" window in
Isaac. A push adds a world-frame velocity kick in a random direction, at the edge of the training
push range (`events.push_robot`) times `--push_scale`, plus a random yaw kick. In Isaac, Space is
normally Kit's Play/Pause hotkey; `play.py` unbinds it for the session.

---

## Export and deploy

Export, verify and gate a policy, then copy the bundle to the robot:

```bash
python -m src.isaaclab.export_onnx --checkpoint checkpoints/<run>/step_<N>.pt
python -m src.isaaclab.verify_onnx_bundle --bundle exported/<bundle>
python -m src.mujoco.policy_gate --bundle exported/<bundle>
```

### `isaaclab/export_onnx.py`

Export a trained SAC actor to a self-contained bundle that needs only `onnxruntime` and `numpy`. It
rebuilds the actor from the checkpoint's own tensor shapes (an MLP actor, plain or mirror-equivariant; a transformer, CNN or LSTM actor cannot be exported)
and then validates the result: 50-dim frame, 12 actions, 50 Hz, IMU gravity sign, and ONNX vs PyTorch
output. It fails if the checkpoint's `step_<N>_obs_scalers.pt` is missing, which is why
`train.normalize_obs` must be on.

```bash
python -m src.isaaclab.export_onnx --checkpoint checkpoints/<run>/step_<N>.pt
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | newest | `step_<N>.pt` to export |
| `--out_dir` | `exported/<run>_step_<N>/` | Bundle folder |
| `--config_path` | the run's `config.yaml` | Config to read instead of the run's own (history size, action map, servo model, gait clock, command limit) |
| `--opset` | `17` | ONNX opset |
| `--action_map` | from the config | `absolute`, `centered` or `residual`. Only needed when the run's `config.yaml` is missing |

Bundle contents:

| File | Content |
|---|---|
| `policy.onnx` | Observation `[1, 50 × history]` to action `[1, 12]`; the inner normaliser and `tanh` are in the graph |
| `deploy_config.json` | Everything the runtime needs: outer normaliser statistics, observation layout, joint names and limits, action map (home pose and scales for `residual`), servo gain, gait clock, command ranges and limit shape, control rate, tilt fault threshold |
| `checkpoint_info.json` | Provenance: source checkpoint, run, step, time, git commit, network sizes |

### `isaaclab/verify_onnx_bundle.py`

Check a bundle against the repository's own actor code, including the outer normaliser the runtime
applies. Exit code 0 means the bundle reproduces the training-time policy.

```bash
python -m src.isaaclab.verify_onnx_bundle --bundle exported/<bundle>
```

| Argument | Default | Description |
|---|---|---|
| `--bundle` | required | Bundle folder |
| `--checkpoint` | from `checkpoint_info.json` | Checkpoint to compare against |
| `--n` | `200` | Random observations to compare |
| `--tol` | `1e-5` | Maximum allowed difference |

### `mujoco/policy_gate.py`

Pre-deploy gate: runs the exported policy in MuJoCo the way the robot runs it (same action rate
limiter, same zero-filled history start, previous-action slot carrying the applied action) and fails
it on falls, chatter, saturated joints or a bad standing pose. Exit code 0 is PASS, 1 is FAIL.
Backlash is off in this simulator, so it is slightly optimistic.

It reports the offline gain of the policy, closed-loop cases at extra action delays with a zero
command and a forward walk (0.2 m/s), joint saturation, inter-leg contact and foot separation, then a verdict.

```bash
python -m src.mujoco.policy_gate --bundle exported/<bundle>
python -m src.mujoco.policy_gate --bundle exported/<bundle> --seeds 3 --seconds 8 --rate 0.5
```

**Sweep mode** checks per-command behaviour instead (does it walk at 0.1 m/s?). It accepts a bundle
or a raw checkpoint and prints falls, achieved speed, step frequency, swing clearance and a
"walks / stands-stalls" verdict per command:

```bash
python src/mujoco/policy_gate.py --checkpoint checkpoints/<run>/step_<N>.pt --cmds 0.05 0.1 0.2
```

| Argument | Default | Description |
|---|---|---|
| `--bundle` | | Exported bundle folder |
| `--checkpoint` | | Raw checkpoint instead (uses the run's `config.yaml` and observation scaler). Give exactly one of `--bundle` / `--checkpoint` |
| `--cmds` | off | Sweep mode: forward-velocity commands (vy = wz = 0) |
| `--delays` | gate 0 to 3, sweep 0 | Extra action delays in control steps |
| `--trace` | none | Hardware trace `.npz` with a `frame` array, to measure the gain on real states |
| `--seeds` | `3` | Seeds per case |
| `--seconds` | `8` (sweep `10`) | Episode length |
| `--rate` | `0.5` | Rate limiter: maximum `abs(delta action)` per step (0 = off) |
| `--cases` | `all` | `all`, `cmd0` or `walk`: restrict the closed-loop cases |
| `--legacy-imu` | off | Old Isaac accelerometer convention |
| `--legacy-prev-slot` | off | The policy sees the delayed action in its previous-action slot |

---

## Get-up policy

A separate fall-recovery policy (`config/chhaya_getup.yaml`; the design is in that file's header and in
`envs/isaaclab/getup_env.py`). Its outputs never mix with walking runs: `runs_getup/`, `checkpoints_getup/`,
`exported_getup/`, `data_getup/`.

```bash
# 1. fall bank: limp-settled falls captured with the runtime's settle rule (+ mirrors, + a held-out gate split)
python src/isaaclab/make_fall_bank.py --config_path config/chhaya_getup.yaml --headless   # -> data_getup/fall_bank.npz
# 2. train / smoke test
python -m src.isaaclab.train --task ChhayaGetUp --config_path config/chhaya_getup.yaml --headless
python -m src.isaaclab.train --task ChhayaGetUp --config_path config/chhaya_getup.yaml --num_envs 64
# 3. watch (newest run in checkpoints_getup/ when --checkpoint is omitted); one start category, or the gate split
python -m src.isaaclab.play --task ChhayaGetUp --start_category supine [--bank_split gate]
# 4. export (format 7, policy_role getup) -> exported_getup/<run>_<step>/
python -m src.isaaclab.export_onnx --checkpoint checkpoints_getup/<run>/step_<N>.pt
# 5. MuJoCo gate, including the hand-off to a walking bundle -> data_getup/gate/<bundle>.json
python src/mujoco/getup_gate.py --bundle exported_getup/<name> --walk_bundle exported/<walk name>
```

`mujoco/policy_gate.py` refuses a get-up bundle (it gates walking). A get-up bundle needs a runtime that knows
format 7 (`start_protocol`, `settle`, `handoff`); a walking-only runtime refuses it by version.

## Diagnose

### `isaaclab/check_actuator.py`

Startup checks for the servo model and the action map. Run it after changing the actuator, the action
map or the sim-to-real settings, before spending a training run. It builds the environment like
`train.py`, holds every robot still under a zero command, and prints PASS or FAIL for: the actuator
class and parameter file, action `0` reproducing the default pose, the sign of the mount offset, the
external-torque observer, the resting foot height, the sampled servo population, and staying upright.

```bash
python src/isaaclab/check_actuator.py --headless
python src/isaaclab/check_actuator.py --headless --num_envs 64 --seconds 4 --config_path config/my_run.yaml
```

| Argument | Default | Description |
|---|---|---|
| `--config_path` | `config/chhaya.yaml` | Config to check |
| `--num_envs` | `16` | Robots |
| `--seconds` | `3.0` | Simulated time |
| `--no-backlash` | off | Disable backlash (A/B) |
| `--tau-ext` | config | `observer` or `gravity`: override the external-torque source |
| `--friction-scale` | none | Force every servo's friction scale (0 = friction off) |
| `--legacy-dcmotor` | off | Baseline with the pre-BAM motor model; reports standing metrics only |

### `isaaclab/jitter_probe.py`

Is the chatter in the network or in the simulator? It runs a checkpoint deterministically (headless,
flat ground, one fixed command) and reports per-joint chatter numbers; toggle the suspects to see what
each contributes.

```bash
python src/isaaclab/jitter_probe.py --task Chhaya --checkpoint checkpoints/<run>/step_<N>.pt
python src/isaaclab/jitter_probe.py --checkpoint ... --no-backlash --no-action-noise --alpha 0.2
```

| Metric | Meaning |
|---|---|
| `da_rms` | RMS change of the action per step on the `[-1, 1]` scale (network chatter) |
| `dq_hf` | Share of joint-velocity power above 8 Hz (audible buzzing) |
| `sat` | Share of servo steps at the duty ceiling |
| `wb_rms` | Base angular velocity RMS (body wobble) |
| `vx` | Mean forward speed reached |

| Argument | Default | Description |
|---|---|---|
| `--task` | `Chhaya` | Robot |
| `--checkpoint` | required | Checkpoint (SAC) |
| `--config_path` | run's `config.yaml` | Config |
| `--num_envs` | `16` | Robots |
| `--seconds` | `12.0` | Simulated time (the first 2 s are discarded as settling) |
| `--cmd VX VY WZ` | `0.1 0.0 0.0` | Fixed command |
| `--no-backlash` | off | Disable backlash |
| `--no-action-noise` | off | Disable action noise |
| `--alpha` | config | Override the action low-pass filter coefficient |
| `--legacy-imu` | off | Old accelerometer convention |

### `isaaclab/profile_train.py`

Where does one training iteration spend its time? It builds the environment and SAC like `train.py`
and runs the iteration body in four passes: a baseline rate, wall time per phase, host-device sync
sites, and a `torch.profiler` summary. It is small on purpose (flat terrain, a 100k replay, no
checkpoints), so it can run next to a live training run. Launch costs do not scale with `--num_envs`,
so measure two sizes and extrapolate.

```bash
python -m src.isaaclab.profile_train --task Chhaya --config_path config/chhaya.yaml \
    --num_envs 1024 --out profile_out
```

| Argument | Default | Description |
|---|---|---|
| `--task` | `Chhaya` | Robot |
| `--config_path` | `config/chhaya.yaml` | Config (must use `model: sac`) |
| `--num_envs` | `1024` | Robots |
| `--warmup` | `60` | Iterations before any measurement |
| `--baseline_iters` | `100` | Iterations for the baseline pass |
| `--phase_iters` | `100` | Iterations for the per-phase timing |
| `--sync_iters` | `3` | Iterations for sync-site detection |
| `--profiler_iters` | `10` | Iterations under `torch.profiler` |
| `--buffer_size` | `100000` | Replay size |
| `--rough` | off | Keep the config's rough terrain (the first reset is slow) |
| `--out` | none | Folder for the scratch logs, trace and JSON |
