from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd  # type: ignore

from alg.ppo.model import Agent
from alg.ppo.envs import make_vec_envs


def evaluate(
    agent: Agent,
    obs_rms: RunningMeanStd,
    env_name: str,
    num_processes: int,
    device: torch.device,
    min_num_episodes: int = 1,
    seed: Optional[int] = None,
    **env_kwargs: Optional[Dict[str, Any]],
) -> float:
    """
    モデルの性能を評価する

    Parameters
    ----------
    agent : Agent
        評価するエージェント
    obs_rms : RunningMeanStd
        観測値の平均と標準偏差
    env_name : str
        環境名
    num_processes : int
        プロセス数
    device : torch.device
        デバイス
    min_num_episodes : Optional[int]
        評価するエピソード数, by default 1
    seed : Optional[int]
        シード, by default None

    Returns
    -------
    float
        平均収益
    """

    envs = make_vec_envs(env_name, num_processes, None, device, training=False, seed=seed, **env_kwargs)
    envs.obs_rms = obs_rms

    episode_rewards: List[float] = []

    obs = envs.reset()

    while len(episode_rewards) < min_num_episodes:

        with torch.no_grad():
            _, actions, _ = agent.act(obs, deterministic=True)

        obs, _, _, infos = envs.step(actions)

        for info in infos:
            if "episode" in info.keys():
                episode_rewards.append(info["episode"]["r"])

    envs.close()

    return float(np.mean(episode_rewards))
