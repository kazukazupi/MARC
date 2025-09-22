from typing import Any, List

from envs.base import MultiAgentEvoGymBase
from envs.box_push import AboveBoxPushEnvClass, BoxPushEnvClass
from envs.chimney_clash import ChimneyClashEnvClass
from envs.pass_and_block import PassAndBlockEnvClass
from envs.sumo import SumoEnvClass

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
