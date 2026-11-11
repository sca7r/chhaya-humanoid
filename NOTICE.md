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

- `hardware/upper_body/`: new. Arms, claws, chest and head: parametric CAD source, STEP and STL
  files, checking tools, documentation.
- `hardware/robot_description/`: added `urdf/Robot_upper_sts3215.urdf`, `meshes/upper_sts3215/`
  and `upper_sts3215_mass.json`. The original files are untouched.
- `training/`: upper-body support, off unless a config enables it. New files:
  `config/config_upper_body.yaml`, `envs/assets/robotV2/biped_robot_upper.py`, `robot_upper.usd`,
  `robot_v2_upper_sts3215.xml`, `envs/isaaclab/mdp/arm_motion.py`, `upper_body/`, two test files,
  `docs/upper_body/`. Modified files: `envs/isaaclab/biped_env.py`, `envs/isaaclab/biped_env_cfg.py`,
  `envs/mujoco/biped_env_v2.py`, `src/isaaclab/env_config.py`, `src/mujoco/policy_gate.py`,
  `config/schema.py`, `.gitignore`.
- `training/robot_hardware` is a link to `../hardware` instead of a submodule.

The legs, electronics and their documentation are the original design.

## Licence of the new work

New files under `training/` are released under the MIT licence, and new files under `hardware/`
under CERN-OHL-P-2.0, the same as the directories they live in.
