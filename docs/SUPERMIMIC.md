# SuperMimic for Chhaya (added 2026-11-18)

A third way to train Chhaya, next to SAC (`train.py`, unchanged and still the default). It follows
**SuperMimic** (Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath, CoRL 2026,
[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)): learn one differentiable **world model** of
the simulator from logged transitions, then improve the policy by backpropagating the return of short
**imagined** rollouts through that model, with a critic for what lies beyond the horizon. In the
paper this trained a G1 whole-body motion tracker on one H100 in 3 h, against 128 GPUs for 7 days for
the PPO baseline.

No reference code had been published when this was written; `training/supermimic/` is a
reimplementation from the paper.

## Run

```bash
cd training
# RTX 3060 laptop (6 GB)
python src/isaaclab/train_supermimic.py --config_path config/chhaya.yaml --num_envs 1024 --preset laptop --minutes 120
# 24 GB or larger GPU: the paper's settings
python src/isaaclab/train_supermimic.py --config_path config/chhaya.yaml --num_envs 4096 --preset paper --minutes 180
# Chhaya Lite
python src/isaaclab/train_supermimic.py --config_path config/chhaya_lite.yaml --num_envs 1024 --preset laptop
tensorboard --logdir runs
```

What to watch: `wm/state` (world-model error, normalised units) should fall below about 0.1;
`env/episode_length` should climb towards 1000 (20 s) and `env/episode_return` should rise.
`policy/imagined_reward` is what the policy believes it earns; if it rises while the real return does
not, the policy is exploiting world-model errors.

## How the code maps to the paper

| Paper | Here |
|---|---|
| Training loop, Fig. 2 / Algorithm A.1 | `learner.py`, `SuperMimic.train`: collect, fit the model, improve the policy, every iteration |
| World model f, reward, termination heads, Eq. 1, k = 8 autoregressive windows | `world_model.py` |
| Imagined rollouts H = 32, survival weight, critic bootstrap, Eq. 2 | `SuperMimic.imagine` |
| Critic on GAE(λ) targets of the imagined steps, Eq. 3 | `SuperMimic.improve_policy` |
| Sensor noise re-added to predicted observations | `Layout.obs_noise_std` (the env's own noise vector) |
| Sim context ξ (domain randomisation values) | `isaac_adapter.py`, `ChhayaAdapter.context` |
| Hyperparameters, Table A.5 | `config.py` (`paper`, `laptop`, `tiny` presets) |

## Differences from the paper

| | Paper | Here | Why |
|---|---|---|---|
| Task | tracking human motion clips; velocity commands only through an encoder trained jointly with the tracker | velocity commands directly, with our existing walking reward | there is no motion library for Chhaya. **The paper does not show velocity walking learned from scratch this way**; this is the main unknown |
| Reward | BeyondMimic tracking reward | our reward (56 terms), learned by the reward head | many terms depend on bookkeeping the state does not hold, so the head learns their average |
| Terrain, pushes | flat, no pushes | flat, no pushes by default (`--rough`, `--pushes` to keep them) | a deterministic model cannot predict a random push |
| State | full simulator state | critic observation (proprio frame + 10 privileged) + true joint angles and velocities, angular velocity, gravity, leg actions two steps back (142 values) | what our env exposes; the servo model's internal state (gear-play position, delay buffers) stays hidden |
| Known slots | — | previous-action slots set to the action, held command, shifted action history: exact, not learned | removes avoidable model error |
| Command | — | held constant over the 32 imagined steps | the env's heading control changes the turn command as the robot yaws; yaw is not in the state |
| Policy | encoder–decoder, 10 frames | our 5-frame MLP actor, no mirror symmetry | keeps the observation the on-robot runtime expects |
| Observation noise in imagination | known noise | the env's Gaussian noise; the vibration noise is not re-added | |
| Reset sampling | return-based, over motion clips | ordinary env resets | no clips |
| Update-to-data ratio 0.4 | not defined in detail | 2.5 env steps per iteration, and at least 2048 new transitions per iteration | my reading; the floor keeps small runs near the paper's data per update (see below) |
| Normalisers | not stated | running statistics from all collected data (sample count capped at 2 M so they keep following the policy); inputs clamped to ±10 | normalisers fixed after a warm-up went stale and broke learning in the tests |
| Guards | none stated | predicted reward kept within the range the simulator has paid; critic values and targets kept within the possible return range | the first Chhaya run diverged without them (critic loss 5 × 10⁵) |
| Initial policy | not stated | last layer scaled to ~0, so the untrained policy holds the home pose | early data of standing, not of falling |
| Executed action | the noisy action | the action the env actually applied, after its own action-noise randomisation | the stored action must match the state the env reports |

## Tested so far (CPU, no GPU)

- **Unit tests** (`tests/test_supermimic.py`): exact action slots and their gradient; replay windows
  and observation history across episode ends; and a small velocity-tracking task where the policy
  learns through the learned model: mean reward per step 0.07–0.09 → 0.98 in 400 iterations on four
  seeds (1.0 = perfect; the exact dynamics in place of the model reach 0.95).
- **Chhaya on Isaac Lab 3.0**: see the next section.

What the toy taught: the method fails when the normalisers no longer describe the data (a position
that grows without bound, or a policy that moves into states the warm-up never saw): the model's
error grows and the critic extrapolates. Hence the running normalisers and the guards. Chhaya's state
has no unbounded value on flat ground; on rough terrain the root height is a world height, one more
reason to start flat.

## Chhaya on a CPU (4 robots, flat ground, all randomisation, `tiny` preset)

| Check | Result |
|---|---|
| Interface | state 142 values, action 20, sim context 37, observation 5 × 74; the exact action slots match the env to 0.0 |
| Doing nothing (zero action), 8 robots, 5 s | all stand; also with the exploration noise added |
| Training, 600 iterations, learning rate 1e-3 | stable after the guards were added; world-model error 0.24 → 0.12; episodes 30–55 steps |
| Training, 600 iterations, paper learning rate 2e-4 | world-model error 0.51 → 0.14; first episodes 154 steps, later 25–75 steps; critic loss rising to ~5000 with its estimate pinned at the upper bound |
| Speed | about 3 iterations per second |

**The policy ended worse than standing still.** Each iteration gathered 10 new transitions against about
10,000 in the paper, so the policy was updated roughly 1,000 times more often per sample, against a
model trained on about 6,000 transitions in total. The policy and critic then optimise the model's
and the critic's errors. That is why `min_transitions_per_iter` (2048) now exists; with 1024 robots and
the default 2.5 steps per iteration the floor does not change anything. Whether SuperMimic learns to
walk is therefore still open: it needs a GPU run in the paper's regime.

## First GPU run: what to do

1. Laptop, 1024 robots, `--preset laptop --minutes 120`. Check that `wm/state` falls below about 0.1
   and `env/episode_length` rises. If the episode length drops below its first values and stays there
   while `critic/loss` climbs, the policy is overfitting the model: stop and report the logs.
2. Run SAC on the same config for the same wall time and compare tracking reward and episode length.

## Not done yet

- **Not run on a GPU**, so no walking result. Run SAC and SuperMimic on the same config and compare
  `env/episode_length` and tracking reward against wall-clock time.
- **Play and ONNX export** read SAC checkpoints only. A SuperMimic checkpoint (`supermimic_*.pt`) holds
  the actor with its observation normalisation, but `play.py` and `export_onnx.py` do not load it yet.
- Return-based resets, rough terrain and pushes.
- Critic grounded on real returns from the replay (not in the paper); worth trying if the critic
  runs away on the GPU too.
