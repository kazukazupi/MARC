import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from evogym import get_full_connectivity, sample_robot
from stable_baselines3.common.running_mean_std import RunningMeanStd

from alg.ppo import Agent
from envs import make_vec_envs


def evaluate(
    agents: List[Optional[Agent]],
    obs_rms_dict: Dict[str, Optional[RunningMeanStd]],
    env_name: str,
    num_processes: int,
    device: torch.device,
    min_num_episodes: int = 1,
    seed: Optional[int] = None,
    **env_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, float]:

    if all([obs_rms is None for obs_rms in obs_rms_dict.values()]):
        norm_obs, norm_reward = False, False
    else:
        norm_obs, norm_reward = True, True

    envs = make_vec_envs(
        env_name,
        num_processes,
        None,
        device,
        training=False,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        seed=seed,
        **env_kwargs,
    )

    assert len(agents) == len(envs.agents), "The number of agents must be equal to the number of environments."
    assert len(obs_rms_dict) == len(envs.agents), "The number of obs_rms must be equal to the number of environments."

    for a in envs.agents:
        if obs_rms_dict[a] is not None:
            envs.obs_rms_dict[a] = obs_rms_dict[a]

    episode_rewards: Dict[str, List[float]] = {a: [] for a in envs.agents}

    observations = envs.reset()

    while len(episode_rewards[envs.agents[0]]) < min_num_episodes:

        # set actions
        actions = {}
        for a, agent in zip(envs.agents, agents):
            if agent is not None:
                with torch.no_grad():
                    _, action, _ = agent.act(observations[a], deterministic=True)
                actions[a] = action
            else:
                actions[a] = torch.Tensor(np.array([envs.action_space(a).sample() for _ in range(num_processes)]))

        # step
        observations, _, _, infos = envs.step(actions)

        for a, info in infos.items():
            for env_idx in range(num_processes):
                if "episode" in info[env_idx]:
                    episode_rewards[a].append(info[env_idx]["episode"]["r"])

    assert all(
        [len(episode_rewards[a]) == len(episode_rewards[envs.agents[0]]) for a in envs.agents]
    ), "The number of episodes must be equal for all agents."

    return {a: float(np.mean(episode_rewards[a])) for a in envs.agents}


if __name__ == "__main__":

    body_1 = np.array(
        [
            [0, 0, 0, 2, 3],
            [2, 0, 4, 4, 4],
            [1, 3, 2, 0, 1],
            [1, 1, 1, 3, 4],
            [0, 3, 0, 2, 0],
        ]
    )
    connections_1 = get_full_connectivity(body_1)

    body_2 = np.fliplr(body_1)
    connections_2 = get_full_connectivity(body_2)

    env_name = "Sumo-v0"

    agent_names = ["robot_1", "robot_2"]
    agents: List[Optional[Agent]] = []
    obs_rms_dict = {}
    logs = {"robot_1": "log4", "robot_2": "log4"}

    for a in agent_names:
        param, obs_rms = torch.load(os.path.join(logs[a], a, "controller.pt"))
        obs_dim = param["base.actor.0.weight"].shape[1]
        action_dim = param["dist.fc_mean.bias"].shape[0]
        agent = Agent(obs_dim=obs_dim, hidden_dim=64, action_dim=action_dim)
        agent.load_state_dict(param)
        agents.append(agent)
        obs_rms_dict[a] = obs_rms

    seed = 16
    # np.random.seed(seed)
    # random.seed(seed)

    # body_1, connections_1 = sample_robot((5, 5))
    # body_2, connections_2 = sample_robot((5, 5))

    # env_name = "Sumo-v0"

    # agents: List[Optional[Agent]] = [None, None]
    # obs_rms_dict = {"robot_1": None, "robot_2": None}

    env_kwargs = {
        "body_1": body_1,
        "body_2": body_2,
        "connections_1": connections_1,
        "connections_2": connections_2,
        "render_mode": "human",
    }

    returns = evaluate(
        agents,
        obs_rms_dict,
        env_name,
        num_processes=1,
        device=torch.device("cpu"),
        min_num_episodes=1,
        seed=seed,
        **env_kwargs,
    )

    print(returns)
