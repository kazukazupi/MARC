import argparse
import logging
import math
import os
from typing import Dict, List

import torch

from alg.coea.coea_utils import get_matches, get_percent_survival_evals, load_evo_metadata, save_evo_metadata
from alg.coea.population import Population
from alg.ppo import train
from evaluate import evaluate
from utils import get_agent_names, load_args, save_args


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.env_name, args.exp_dirname)
    metadata_dir_path = os.path.join(save_path, "metadata")
    log_file = os.path.join(save_path, "experiment.log")
    agent_names = get_agent_names()

    if not args.is_continue:

        os.makedirs(save_path)
        os.mkdir(metadata_dir_path)

        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s",
        )
        logging.info(f"Starting experiment at {save_path}")

        save_args(args, metadata_dir_path)
        save_evo_metadata(metadata_dir_path, 0)

        populations = {
            name: Population(name, os.path.join(save_path, name), args.pop_size, args.robot_shape)
            for name in agent_names
        }
        num_trainings = 0

    else:
        assert os.path.exists(save_path), "Experiment directory does not exist"
        assert os.path.exists(metadata_dir_path), "Metadata directory does not exist"
        assert os.path.exists(log_file), "Log file does not exist"

        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s",
        )
        logging.info(f"Continuing experiment at {save_path}")

        args = load_args(metadata_dir_path)
        num_trainings = load_evo_metadata(metadata_dir_path)
        populations = {
            name: Population(name, os.path.join(save_path, name), args.pop_size, args.robot_shape, is_continuing=True)
            for name in agent_names
        }

    while True:
        generation = populations[agent_names[0]].generation
        logging.info(f"# Start Training (Generation {generation})")

        # train
        matches = get_matches(
            populations[agent_names[0]].get_training_indices(),
            populations[agent_names[1]].get_training_indices(),
            1,
            agent_names,
            metadata_dir_path,
            generation,
            "train",
        )

        for match in matches:
            if num_trainings >= args.max_trainings:
                break
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}
            train(args, structures)
            logging.info(
                f"Trained {match[agent_names[0]]} vs {match[agent_names[1]]} ({num_trainings+1}/{args.max_trainings})"
            )
            num_trainings += 1
            save_evo_metadata(metadata_dir_path, num_trainings)

        # evaluate
        logging.info(f"# Start Evaluation (Generation {generation})")
        matches = get_matches(
            populations[agent_names[0]].get_evaluation_indices(),
            populations[agent_names[1]].get_evaluation_indices(),
            args.eval_num_opponents,
            agent_names,
            metadata_dir_path,
            generation,
            "eval",
        )

        for match in matches:
            structures = {agent_name: populations[agent_name][id] for agent_name, id in match.items()}

            if structures[agent_names[0]].has_fought(match[agent_names[1]]):
                assert structures[agent_names[1]].has_fought(match[agent_names[0]])
                logging.info(f"Skipped evaluation {match[agent_names[0]]} vs {match[agent_names[1]]}")
                continue

            results = evaluate(
                structures,
                args.env_name,
                num_processes=1,
                device=torch.device("cpu"),
                min_num_episodes=1,
                seed=None,
            )

            for agent_name in agent_names:
                opponent_name = agent_names[1] if agent_name == agent_names[0] else agent_names[0]
                populations[agent_name].set_score(match[agent_name], match[opponent_name], results[agent_name])
            logging.info(
                f"Evaluated {match[agent_names[0]]}({results[agent_names[0]]:.3f}) "
                f"vs {match[agent_names[1]]}({results[agent_names[1]]:.3f})"
            )

        for name in agent_names:
            populations[name].dump_fitnesses()

        if num_trainings >= args.max_trainings:
            break

        # selection, reproduction
        logging.info(f"# Start Selection, Reproduction (Generation {generation})")
        percent_survival = get_percent_survival_evals(num_trainings, args.max_trainings)
        num_survivors = max(2, math.ceil(args.pop_size * percent_survival))
        num_reproductions = min(args.pop_size - num_survivors, args.max_trainings - num_trainings)
        non_survivors_dict: Dict[str, List[int]] = {}
        logging.info(f"Percent survival: {percent_survival * 100:.2f}%")
        logging.info(f"Num survivors: {num_survivors}")
        logging.info(f"Num reproductions: {num_reproductions}")
        for name in agent_names:
            non_survivors = populations[name].update(num_survivors, num_reproductions)
            non_survivors_dict[name] = non_survivors
        for agent_name in agent_names:
            opponent_name = agent_names[1] if agent_name == agent_names[0] else agent_names[0]
            for opponent_id in non_survivors_dict[opponent_name]:
                populations[agent_name].delete_score(opponent_id)

    logging.info("Experiment finished")
