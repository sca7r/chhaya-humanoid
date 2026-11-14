"""Smoke test: does Chhaya run in Isaac Lab 3.0 without Isaac Sim (Newton backend, MuJoCo-Warp solver)?

    HEADLESS=1 python smoke_test.py --usd $PWD/../../envs/assets/robotV2/chhaya_newton.usd [--device cuda:0 --num_envs 1024]

Runs on the CPU too (slow), so it needs no GPU. Expected: 20 joints, 21 bodies, 2.567 kg, no falls, base
height ~0.28 m, vertical contact force = weight (25.2 N). Not Chhaya's training environment (that check is env_check.py): a minimal
DirectRLEnv with plain PD joints, to check the asset, terrain, contact sensor and API.

Spawns chhaya.usd as an Articulation on generated rough terrain with a contact sensor, in a
DirectRLEnv, and holds the home pose with PD joint targets. Reports what loaded and how it behaves.
"""
import argparse, sys, time, math
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--num_envs", type=int, default=2)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--terrain", default="rough")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
import isaaclab.terrains as terrain_gen
FLAT_GEN = terrain_gen.TerrainGeneratorCfg(size=(8.0, 8.0), border_width=2.0, num_rows=1, num_cols=1, sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)})
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

ARM_REST = {"l_shoulder_pitch": 0.0, "r_shoulder_pitch": 0.0, "l_shoulder_roll": 0.20, "r_shoulder_roll": 0.20,
            "l_arm_yaw": 0.0, "r_arm_yaw": 0.0, "l_elbow": -0.35, "r_elbow": -0.35}

ROBOT = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=args.usd, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.31), joint_pos={"(l|r)_(hip|knee|ankle).*": 0.0, **ARM_REST}, joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=20.0, damping=0.8,
                                          joint_effort_limit=2.94, armature=0.01)},
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(num_rows=2, num_cols=2, curriculum=False) if args.terrain == "rough" else FLAT_GEN,
        max_init_terrain_level=None,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot: ArticulationCfg = ROBOT
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class EnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 20
    observation_space = 40
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, render_interval=4, device=args.device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=300, cone="pyramidal", impratio=1.0,
                                       integrator="implicitfast", use_mujoco_contacts=False),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
            num_substeps=2, default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)))
    scene: SceneCfg = SceneCfg(num_envs=args.num_envs, env_spacing=2.5, replicate_physics=True)


class ChhayaEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self.robot = self.scene["robot"]
        self.default = self.robot.data.default_joint_pos.torch.clone()
        self.actions = torch.zeros(self.num_envs, 20, device=self.device)

    def _pre_physics_step(self, actions):
        self.actions = actions

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.default + 0.1 * self.actions)

    def _get_observations(self):
        d = self.robot.data
        return {"policy": torch.cat([d.joint_pos.torch - self.default, d.joint_vel.torch], dim=-1)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        return z < 0.12, self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        d = self.robot.data
        pose = d.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=d.default_root_vel.torch[env_ids].clone(), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=d.default_joint_pos.torch[env_ids].clone(), env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=d.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids)


cfg = EnvCfg()
with launch_simulation(cfg, argparse.Namespace(device=args.device, viz="none", headless=True)):
    env = ChhayaEnv(cfg)
    r = env.robot
    print("RESULT backend:", type(r).__module__)
    print("RESULT joints:", r.num_joints, r.joint_names)
    print("RESULT bodies:", r.num_bodies, r.body_names)
    m = r.data.default_mass.torch if hasattr(r.data, "default_mass") else r.data.body_mass.torch
    print("RESULT total mass kg:", round(float(m[0].sum()), 4))
    print("RESULT root quat at reset (xyzw):", [round(float(v), 3) for v in r.data.root_quat_w.torch[0]])
    env.reset()
    t = time.time()
    falls = 0
    for k in range(args.steps):
        with torch.inference_mode():
            obs, rew, term, trunc, _ = env.step(torch.zeros(args.num_envs, 20, device=env.device))
        falls += int(term.sum())
    dt = time.time() - t
    z = (r.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).tolist()
    c = env.scene["contact"]
    f = c.data.net_forces_w.torch
    names = c.body_names
    feet = [i for i, n in enumerate(names) if "foot" in n.lower() or "ankle" in n.lower()]
    print("RESULT finite obs:", bool(torch.isfinite(obs["policy"]).all()), "| falls:", falls, "| base height:", [round(v, 3) for v in z])
    print("RESULT contact sensor bodies:", len(names), "| vertical force sum per env [N]:", [round(float(v), 1) for v in f[..., 2].sum(1)],
          "(weight =", round(float(m[0].sum()) * 9.81, 1), "N)")
    print("RESULT air time field:", tuple(c.data.current_air_time.torch.shape))
    print(f"RESULT speed: {args.steps * args.num_envs / dt:.1f} env-steps/s on {env.device} ({args.num_envs} envs)")
    env.close()
