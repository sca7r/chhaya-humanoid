"""Run the real training environment (BipedEnv + config/chhaya.yaml) on Isaac Lab 3.0 / Newton.

    python src/isaaclab/env_check.py --num_envs 2 --steps 200            # CPU works (slow)
    python src/isaaclab/env_check.py --num_envs 1024 --device cuda:0

Holds the home pose (zero action) for the first half, then random actions; prints what the env sees.
"""
import argparse, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (os.path.join(ROOT, "torch_rl_algorithms"), ROOT):
    sys.path.insert(0, p)
import torch
from src.isaaclab.launch import add_launch_args, launch

ap = argparse.ArgumentParser()
ap.add_argument("--config_path", default=os.path.join(ROOT, "config/chhaya.yaml"))
ap.add_argument("--task", default="BipedRobotV2")
ap.add_argument("--num_envs", type=int, default=2)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--no_dr", action="store_true")
ap.add_argument("--flat", action="store_true", help="flat slab instead of the rough mix")
add_launch_args(ap)
args = ap.parse_args()

from utils import Config
from utils.residual_home import freeze_residual_map
from src.isaaclab.env_config import build_env_cfg, describe
from src.isaaclab.tasks import resolve_task

EnvCfg, Env = resolve_task(args.task)
config = Config(args.config_path)
freeze_residual_map(config)
cfg = build_env_cfg(config, EnvCfg, training=not args.no_dr, num_envs=args.num_envs)
cfg.seed = 0
if args.flat:
    from envs.isaaclab.biped_env_cfg import FLAT_TERRAIN_CFG
    cfg.use_rough_terrain = False
    cfg.terrain.terrain_generator = FLAT_TERRAIN_CFG
    cfg.scene.contact_forces.filter_shape_prim_expr = ["/World/ground/.*"]
print("CHECK contract:", describe(cfg))
with launch(cfg, args):
    env = Env(cfg=cfg)
    r = env.robot
    print("CHECK joints:", r.num_joints, "| bodies:", r.num_bodies, "| mass kg:", round(float(env.rd.body_mass[0].sum()), 4))
    print("CHECK contact sensor names:", env.contact_sensor.sensor_names)
    # Which link pairs can collide inside one robot (self-collision), from the Newton model.
    try:
        from isaaclab_newton.physics import NewtonManager
        m = NewtonManager.get_model()
        sb = m.shape_body.numpy(); sw = m.shape_world.numpy(); bl = list(m.body_label)
        flags = m.shape_flags.numpy()
        from newton import ShapeFlags
        ids = [i for i in range(m.shape_count) if sw[i] == 0 and sb[i] >= 0 and flags[i] & int(ShapeFlags.COLLIDE_SHAPES)]
        filt = set(map(tuple, m.shape_collision_filter_pairs)) if not hasattr(m.shape_collision_filter_pairs, "numpy") else set(map(tuple, m.shape_collision_filter_pairs.numpy().tolist()))
        pairs = set()
        for a in ids:
            for b in ids:
                if a < b and sb[a] != sb[b] and (a, b) not in filt and (b, a) not in filt:
                    pairs.add(tuple(sorted((bl[sb[a]].split("/")[-1], bl[sb[b]].split("/")[-1]))))
        allp = len({tuple(sorted((bl[sb[a]].split("/")[-1], bl[sb[b]].split("/")[-1]))) for a in ids for b in ids if a < b and sb[a] != sb[b]})
        print("CHECK self-collision: colliding shapes per robot", len(ids), "| link pairs that can collide", len(pairs), "of", allp)
        want = [("base_link", "l_forearm_link"), ("l_foot_link_1", "r_foot_link_1"), ("l_calf_link_1", "l_foot_link_1"), ("base_link", "l_shoulder_link"), ("l_calf_link_1", "r_calf_link_1")]
        print("CHECK self-collision:", {f"{a}~{b}": (tuple(sorted((a, b))) in pairs) for a, b in want})
    except Exception as e:  # report only
        print("CHECK self-collision: could not read the model:", repr(e)[:200])
    obs, _ = env.reset()
    na = env.cfg.action_space
    print("CHECK obs:", {k: tuple(v.shape) for k, v in obs.items()}, "| actions:", na)
    t0 = time.time(); falls = 0; rew_sum = 0.0; hz = []
    for k in range(args.steps):
        a = torch.zeros(env.num_envs, na, device=env.device)
        if k >= args.steps // 2:
            a = 0.3 * torch.randn_like(a)
        with torch.inference_mode():
            obs, rew, term, trunc, info = env.step(a)
        falls += int(term.sum()); rew_sum += float(rew.mean())
        if k == args.steps // 2 - 1:
            d = env.rd
            z = (d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).tolist()
            f = env.cs.net_forces_w[:, env._feet_ids_t]
            print("CHECK standing: base height", [round(v, 3) for v in z], "| quat wxyz", [round(float(v), 3) for v in d.root_quat_w[0]])
            print("CHECK standing: foot Fz [N]", [[round(float(v), 1) for v in row] for row in f[..., 2]],
                  "| weight", round(float(d.body_mass[0].sum()) * 9.81, 1))
            print("CHECK standing: imu acc", [round(float(v), 2) for v in env.imu_lin_acc_b[0]], "gyro", [round(float(v), 3) for v in env.imu_ang_vel_b[0]],
                  "grav", [round(float(v), 3) for v in env.projected_gravity_b[0]])
            print("CHECK standing: |torque| max", round(float(d.applied_torque.abs().max()), 3), "| falls so far", falls,
                  "| mean reward/step", round(rew_sum / (k + 1), 4))
            nf = torch.norm(env.cs.net_forces_w, dim=-1)
            nf[:, env._feet_ids_t] = 0.0
            names = env.contact_sensor.sensor_names
            print("CHECK standing: largest non-foot contact force [N]", round(float(nf.max()), 2), "on", names[int(nf.max(dim=0)[0].argmax())],
                  "| floor/feet force matrix:", None if env.cs.force_matrix_w_history is None else tuple(env.cs.force_matrix_w_history.shape))
            if env.cs.force_matrix_w_history is not None:
                fm = env.cs.force_matrix_w_history[:, 0, env._feet_ids_t].sum(dim=2)
                print("CHECK standing: floor part of foot force Fz", [[round(float(v), 1) for v in row] for row in fm[..., 2]])
            if env.bam_actuators:
                b = env.bam_actuators[0]
                print("CHECK BAM: inertia_eff", [round(float(v), 4) for v in b.inertia_eff[0][:6]], "| tau_ext", [round(float(v), 3) for v in b.tau_ext[0][:6]])
    dt = time.time() - t0
    ok = all(bool(torch.isfinite(v).all()) for v in obs.values())
    print("CHECK finite obs:", ok, "| falls:", falls, "| mean reward/step:", round(rew_sum / args.steps, 4))
    log = env.extras.get("log", {})
    for k in ("Diagnostics/total_mass_mean_kg", "Diagnostics/com_drift_max_m", "Episode_Termination/base_contact", "Episode_Termination/time_out", "Torque/mean_abs"):
        if k in log:
            print("CHECK", k, round(float(log[k]), 4))
    print("CHECK log keys:", len(log), "| e.g.", [(k, round(float(v), 4)) for k, v in list(log.items())[:6]])
    print(f"CHECK speed: {args.steps * env.num_envs / dt:.1f} env-steps/s on {env.device} ({env.num_envs} envs)")
    env.close()
