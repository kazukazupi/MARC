import argparse
import os

import numpy as np

from alg.coea.matching import get_matches
from alg.coea.population import Population
from alg.ppo import train
from evaluate import evaluate
from utils import get_agent_names


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.env_name, args.exp_dirname)
    os.makedirs(save_path)

    agent_names = get_agent_names()

    populations = {name: Population(os.path.join(save_path, name), args) for name in agent_names}

    while True:

        # train
        matches = get_matches(args.pop_size, 1, agent_names)
        for match in matches:
            print(match)
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}
            train(args, structures)

        # evaluate
        fitnesses = {name: np.zeros(args.pop_size) for name in agent_names}

        matches = get_matches(args.pop_size, args.eval_num_opponents, agent_names)
        for match in matches:
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}
            results = evaluate(
                structures,
                args.env_name,
                num_processes=1,
                device=args.device,
                min_num_episodes=1,
                seed=None,
            )
            print(match, results)

            for name, id in match.items():
                fitnesses[name][id] += results[name]

        fitnesses = {name: fitnesses[name] / args.eval_num_opponents for name in agent_names}
        for name in agent_names:
            populations[name].fitnesses = fitnesses[name]

        break
        # selection

        # reproduce
