import argparse
import logging
import math
import os

import numpy as np
import torch

from alg.coea.coea_utils import get_matches, get_percent_survival_evals
from alg.coea.population import Population
from alg.ppo import train
from evaluate import evaluate
from utils import get_agent_names, save_args


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.env_name, args.exp_dirname)
    os.makedirs(save_path)

    log_file = os.path.join(save_path, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s",
    )
    logging.info(f"Starting experiment at {save_path}")

    save_args(args, os.path.join(save_path, "args.json"))

    agent_names = get_agent_names()

    populations = {
        name: Population(name, os.path.join(save_path, name), args.pop_size, args.robot_shape) for name in agent_names
    }
    num_trainings = 0

    while True:
        logging.info(f"Generation {populations[agent_names[0]].generation}")

        # train
        matches = get_matches(
            populations[agent_names[0]].get_training_indices(),
            populations[agent_names[1]].get_training_indices(),
            1,
            agent_names,
        )

        for match in matches:
            if num_trainings >= args.max_trainings:
                break

            logging.info(
                f"Training {match[agent_names[0]]} vs {match[agent_names[1]]} ({num_trainings+1}/{args.max_trainings})"
            )
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}
            train(args, structures)
            num_trainings += 1

        # evaluate
        fitnesses = {name: np.zeros(args.pop_size) for name in agent_names}

        matches = get_matches(
            populations[agent_names[0]].get_evaluation_indices(),
            populations[agent_names[1]].get_evaluation_indices(),
            args.eval_num_opponents,
            agent_names,
        )

        for match in matches:
            logging.info(f"Evaluating {match[agent_names[0]]} vs {match[agent_names[1]]}")
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}
            results = evaluate(
                structures,
                args.env_name,
                num_processes=1,
                device=torch.device("cpu"),
                min_num_episodes=1,
                seed=None,
            )

            for name, id in match.items():
                fitnesses[name][id] += results[name]

        fitnesses = {name: fitnesses[name] / args.eval_num_opponents for name in agent_names}
        for name in agent_names:
            populations[name].fitnesses = fitnesses[name]

        if num_trainings >= args.max_trainings:
            break

        # selection, reproduction
        percent_survival = get_percent_survival_evals(num_trainings, args.max_trainings)
        num_survivors = max(2, math.ceil(args.pop_size * percent_survival))
        num_reproductions = min(args.pop_size - num_survivors, args.max_trainings - num_trainings)
        logging.info(f"Percent survival: {percent_survival}")
        logging.info(f"Num survivors: {num_survivors}")
        logging.info(f"Num reproductions: {num_reproductions}")
        for name in agent_names:
            populations[name].update(num_survivors, num_reproductions)

    logging.info("Experiment finished")
