from gymnasium.envs.registration import register

from envs.sumo_env import SimpleSumoEnvClass

register(
    id="Sumo-v0",
    entry_point="envs.sumo_env:SimpleSumoEnvClass",
    max_episode_steps=1000,
)
