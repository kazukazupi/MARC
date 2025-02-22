import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from alg.coea.structure import Structure
from alg.ppo import Agent, make_vec_envs
from envs import AgentID
from utils import get_agent_names


def evaluate(
    structures: Dict[AgentID, Structure],
    env_name: str,
    num_processes: int,
    device: torch.device,
    min_num_episodes: int = 1,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    render_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:

    agent_names = get_agent_names()

    envs = make_vec_envs(
        env_name,
        num_processes,
        None,
        device,
        training=False,
        seed=seed,
        body_1=structures[agent_names[0]].body,
        body_2=structures[agent_names[1]].body,
        connections_1=structures[agent_names[0]].connections,
        connections_2=structures[agent_names[1]].connections,
        render_mode=render_mode,
        render_options=render_options,
    )

    agents = {}
    for a, structure in structures.items():
        controller_path = structure.get_latest_controller_path()
        state_dict, obs_rms = torch.load(controller_path, map_location=device)
        agents[a] = Agent.from_state_dict(state_dict)
        envs.obs_rms_dict[a] = obs_rms

    episode_rewards: Dict[str, List[float]] = {a: [] for a in envs.agents}
    observations = envs.reset()

    while len(episode_rewards[envs.agents[0]]) < min_num_episodes:

        # set actions
        actions = {}
        for a, agent in agents.items():
            with torch.no_grad():
                _, action, _ = agent.act(observations[a], deterministic=True)
            actions[a] = action

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
    agent_names = get_agent_names()

    structures = {a: Structure.from_save_path(os.path.join(args.save_path, a)) for a in agent_names}

    returns = evaluate(
        structures,
        env_name,
        num_processes=1,
        device=torch.device("cpu"),
        min_num_episodes=1,
        render_mode="human",
    )
