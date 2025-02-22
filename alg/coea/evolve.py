import argparse
import os
import random
from typing import Dict, Optional, cast

import torch

from alg.coea.matching import get_matches
from alg.coea.population import Population
from alg.ppo import Agent, train
from evaluate import evaluate
from utils import get_agent_names


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.env_name, args.exp_dirname)
    os.makedirs(save_path)

    agent_names = get_agent_names()

    populations = {name: Population(os.path.join(save_path, name), args) for name in agent_names}

    while True:

        # pair robots
        indices = list(range(args.pop_size))
        random.shuffle(indices)

        # train
        matches = get_matches(args.pop_size, 1, agent_names)
        for match in matches:
            print(match)
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}
            train(args, structures)

        # evaluate
        matches = get_matches(args.pop_size, args.eval_num_opponents, agent_names)
        for match in matches:
            agents = {}
            obs_rms_dict = {}
            for agent_name, id_ in match.items():
                latest_controller_path = populations[agent_name][id_].get_latest_controller_path()
                state_dict, obs_rms = torch.load(latest_controller_path, map_location=args.device)
                agents[agent_name] = Agent.from_state_dict(state_dict)
                obs_rms_dict[agent_name] = obs_rms
            results = evaluate(
                cast(Dict[str, Optional[Agent]], agents),
                obs_rms_dict,
                args.env_name,
                num_processes=1,
                device=args.device,
                min_num_episodes=1,
                seed=None,
                body_1=populations[agent_names[0]][match[agent_names[0]]].body,
                body_2=populations[agent_names[1]][match[agent_names[1]]].body,
                connections_1=populations[agent_names[0]][match[agent_names[0]]].connections,
                connections_2=populations[agent_names[1]][match[agent_names[1]]].connections,
            )
            print(match, results)

        break
