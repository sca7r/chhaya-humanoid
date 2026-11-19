from gymnasium.envs.registration import register

register(
    id="MujocoV2-v0",
    entry_point="envs.mujoco.biped_env_v2:BipedEnvV2",
    max_episode_steps=1000,
)
