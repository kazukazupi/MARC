from typing import Any

from envs.base import MultiAgentEvoGymBase
from envs.chase_env import SimpleChaseEnvClass
from envs.ojama_env import OjamaDepth3EnvClass, OjamaDepth4EnvClass, OjamaDepth5EnvClass
from envs.push_env import AboveObjectPushEnvClass, ObjectPushEnvClass, WallPushEnvClass
from envs.sumo_env import SimpleSumoEnvClass

ENV_CLASSES = [
    AboveObjectPushEnvClass,
    SimpleChaseEnvClass,
    SimpleSumoEnvClass,
    ObjectPushEnvClass,
    OjamaDepth3EnvClass,
    OjamaDepth4EnvClass,
    OjamaDepth5EnvClass,
    WallPushEnvClass,
]


def make(env_name: str, **kwargs: Any) -> MultiAgentEvoGymBase:

    for env_cls in ENV_CLASSES:
        if env_name == env_cls.ENV_NAME:
            return env_cls(**kwargs)

    raise ValueError(f"Unknown environment name: {env_name}")
