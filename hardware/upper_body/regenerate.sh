#!/usr/bin/env bash
# Rebuild everything that depends on the upper-body CAD, in order, and stop at the first failure.
#   pip install cadquery trimesh mujoco python-fcl usd-core pillow
#   ./regenerate.sh
set -euo pipefail
cd "$(dirname "$0")"
python cad/upper_body.py              # parts: STEP + STL, servo stand-ins, parts.json
python tools/build_description.py     # URDF + meshes (hardware/robot_description), MJCF (training/envs/assets)
python tools/check_interference.py    # zero overlap, >= 1 mm moving clearance, claw sweep
python tools/strength_check.py        # safety factor >= 3 at every critical section
python tools/validate_model.py        # MJCF and URDF compile
python tools/build_usd.py             # Isaac Lab USD (training/envs/assets/robotV2/robot_upper.usd)
MUJOCO_GL=${MUJOCO_GL:-egl} python tools/render_robot.py
echo "done. Now run, from training/:  pytest tests/test_upper_body_mujoco.py && python -m upper_body.physics_check"
