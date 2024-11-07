from gym.envs.registration import register  # type:ignore

from envs.sumo_env import SimpleSumoEnvClass

register(
    id="Sumo-v0",
    entry_point="envs.sumo_env:SimpleSumoEnvClass",
    max_episode_steps=1000,
)
