import argparse
import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from evogym import get_full_connectivity, sample_robot  # type: ignore
from stable_baselines3.common.running_mean_std import RunningMeanStd  # type: ignore

from alg.ppo import Agent, make_vec_envs
from envs import AgentID


def evaluate(
    agents: Dict[AgentID, Optional[Agent]],
    obs_rms_dict: Dict[AgentID, Optional[RunningMeanStd]],
    env_name: str,
    num_processes: int,
    device: torch.device,
    min_num_episodes: int = 1,
    seed: Optional[int] = None,
    **env_kwargs: Any,
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

    for a, obs_rms in obs_rms_dict.items():
        if obs_rms is not None:
            envs.obs_rms_dict[a] = obs_rms

    episode_rewards: Dict[str, List[float]] = {a: [] for a in envs.agents}

    observations = envs.reset()

    while len(episode_rewards[envs.agents[0]]) < min_num_episodes:

        # set actions
        actions = {}
        for a, agent in agents.items():
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.save_path, "env_info.json"), "r") as f:
        env_info = json.load(f)

    env_name = env_info["env_name"]

    body_1 = np.load(os.path.join(args.save_path, env_info["agents"][0], "body.npy"))
    connections_1 = np.load(os.path.join(args.save_path, env_info["agents"][0], "connections.npy"))

    body_2 = np.load(os.path.join(args.save_path, env_info["agents"][1], "body.npy"))
    connections_2 = np.load(os.path.join(args.save_path, env_info["agents"][1], "connections.npy"))

    agents: Dict[AgentID, Optional[Agent]] = {}
    obs_rms_dict = {}

    for a in env_info["agents"]:
        controller_paths = sorted(glob.glob(os.path.join(args.save_path, a, "controller_*.pt")))
        latest_controller_path = max(controller_paths, key=os.path.getctime)
        print(f"Loading {latest_controller_path}")
        state_dict, obs_rms = torch.load(latest_controller_path, map_location="cpu")
        agents[a] = Agent.from_state_dict(state_dict)
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
