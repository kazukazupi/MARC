from typing import Any, List

from envs.base import MultiAgentEvoGymBase
from envs.chimney_env import ChimneyClashEnvClass
from envs.ojama_env import PassAndBlockEnvClass
from envs.push_env import AboveBoxPushEnvClass, BoxPushEnvClass
from envs.sumo_env import SumoEnvClass

ENV_CLASSES: List[type[MultiAgentEvoGymBase]] = [
    SumoEnvClass,
    BoxPushEnvClass,
    AboveBoxPushEnvClass,
    PassAndBlockEnvClass,
    ChimneyClashEnvClass,
]


def make(env_name: str, **kwargs: Any) -> MultiAgentEvoGymBase:

    for env_cls in ENV_CLASSES:
        if env_name == env_cls.ENV_NAME:
            return env_cls(**kwargs)

    raise ValueError(f"Unknown environment name: {env_name}")
