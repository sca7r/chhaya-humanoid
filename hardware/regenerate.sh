#!/usr/bin/env bash
# Rebuild everything that depends on the parametric CAD (chest, head, arms, claws), in order,
# and stop at the first failure. The leg parts are Fusion exports and are not regenerated.
#   pip install cadquery trimesh mujoco python-fcl usd-core pillow
#   ./regenerate.sh
set -euo pipefail
cd "$(dirname "$0")"
python cad/parametric/chhaya_cad.py   # parts: STEP (cad/step/bodies) + STL (print/stl), servo stand-ins
python tools/build_description.py     # chhaya.urdf + meshes (robot_description), chhaya.xml (training/envs/assets)
python tools/check_interference.py    # zero overlap, >= 1 mm moving clearance, claw sweep
python tools/strength_check.py        # safety factor >= 3 at every critical section
python tools/validate_model.py        # MJCF and URDF compile
python tools/build_usd.py             # Isaac Lab USD (training/envs/assets/robotV2/chhaya.usd)
MUJOCO_GL=${MUJOCO_GL:-egl} python tools/render_robot.py
echo "done. Now run, from training/:  pytest tests/test_chhaya_mujoco.py && python -m chhaya.physics_check"
