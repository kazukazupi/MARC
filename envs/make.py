from typing import Any

from envs.base import MultiAgentEvoGymBase
from envs.chimney_env import ChimneyClashEnvClass
from envs.ojama_env import OjamaDepth4EnvClass
from envs.push_env import AboveObjectPushEnvClass, ObjectPushEnvClass
from envs.sumo_env import SimpleSumoEnvClass

ENV_CLASSES = [
    AboveObjectPushEnvClass,
    ChimneyClashEnvClass,
    SimpleSumoEnvClass,
    ObjectPushEnvClass,
    OjamaDepth4EnvClass,
]


def make(env_name: str, **kwargs: Any) -> MultiAgentEvoGymBase:

    for env_cls in ENV_CLASSES:
        if env_name == env_cls.ENV_NAME:
            return env_cls(**kwargs)

    raise ValueError(f"Unknown environment name: {env_name}")
