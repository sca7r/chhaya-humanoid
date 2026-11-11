# Notice and credits

Chhaya is built on **Sir Servington**, the open-source biped by **Pablo Gomez Martinez**
(GitHub: AsterisCrack). Without that work there would be no Chhaya: the legs, the electronics,
the servo model and the whole training stack are his.

| Part of this repository | Source | Licence |
|---|---|---|
| `training/` | https://github.com/AsterisCrack/BipedRobot at commit `e0705b9` | MIT (`training/LICENSE.txt`), © 2025 Pablo Gomez Martinez |
| `hardware/` | https://github.com/AsterisCrack/BipedRobotHardware at commit `90902ec` | CERN-OHL-P-2.0 (`hardware/LICENSE`); third-party notices inside it are unchanged (e.g. `hardware/robot_description/LICENSE-fusion2urdf`) |
| `training/torch_rl_algorithms`, `training/robot_motion_reference` | submodules, pinned to the commits upstream uses | their own licences |

The first two commits of this repository are those sources, unmodified, so `git diff` against
them shows exactly what Chhaya changes.

## What Chhaya changes (the modification notice CERN-OHL-P-2.0 asks for)

Made in November 2026.

- `hardware/`: added the chest, head, arms and claws: parametric CAD source
  (`cad/parametric/`), their STEP files in `cad/step/bodies/` and print files in `print/stl/`
  (the files not present in the source commit), `tools/`, `regenerate.sh`, `docs/design.md`,
  `docs/build/04_chest_arms_head.md`, renders in `media/chhaya/`.
- `hardware/robot_description/`: added `urdf/chhaya.urdf`, `chhaya_mass.json` and the meshes of
  the generated links (`chest_link*`, `*_shoulder_link*`, `*_upper_arm_link*`, `*_elbow_link*`,
  `*_forearm_link*`). The original URDF and meshes are untouched.
- `hardware/README.md`, `bom/`, `cad/README.md`, `print/README.md`, `robot_description/README.md`,
  `docs/build/` (steps renumbered so the chest, arms and head follow the legs): edited to describe
  the whole robot (20 servos, added parts and costs).
- `training/`: support for the arms. New files: `config/chhaya.yaml`,
  `envs/assets/robotV2/chhaya_robot.py`, `chhaya.usd`, `chhaya.xml`,
  `envs/isaaclab/mdp/arm_motion.py`, `chhaya/`, `tests/test_arm_motion.py`,
  `tests/test_chhaya_mujoco.py`. Modified files: `envs/isaaclab/biped_env.py`,
  `envs/isaaclab/biped_env_cfg.py`, `envs/mujoco/biped_env_v2.py`, `src/isaaclab/env_config.py`,
  `src/mujoco/policy_gate.py`, `config/schema.py`, `.gitignore`. Without an `arms` block in the
  config the code behaves as the original.
- `training/robot_hardware` is a link to `../hardware` instead of a submodule.
- `docs/` at the repository root is new.

The legs, hips, torso, electronics, servo model and training method are the original design.

## Licence of the new work

New files under `training/` are released under the MIT licence, and new files under `hardware/`
under CERN-OHL-P-2.0, the same as the directories they live in.
