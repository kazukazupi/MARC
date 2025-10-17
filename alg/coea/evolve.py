import argparse
import logging
import math
import os
from typing import Dict, List, Union, cast

import torch

from alg.coea.coea_utils import get_matches, get_percent_survival_evals, load_evo_metadata, save_evo_metadata
from alg.coea.population import Population
from alg.coea.structure import DummyRobotStructure, Structure
from alg.ppo import evaluate, train
from utils import AGENT_1, AGENT_2, AGENT_IDS, AgentID, get_opponent_id, load_args, save_args


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.env_name, args.exp_dirname)
    metadata_dir_path = os.path.join(save_path, "metadata")
    log_file = os.path.join(save_path, "experiment.log")

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
            for name in AGENT_IDS
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
            for name in AGENT_IDS
        }

    while True:
        generation = populations[AGENT_1].generation
        logging.info(f"# Start Training (Generation {generation})")

        # train
        matches = get_matches(
            populations[AGENT_1].get_training_indices(),
            populations[AGENT_2].get_training_indices(),
            1,
            metadata_dir_path,
            generation,
            "train",
        )

        for match in matches:
            if num_trainings >= args.max_trainings:
                break
            structures = {agent_id: populations[agent_id][id] for agent_id, id in match.items()}
            train(args, cast(Dict[AgentID, Union[Structure, DummyRobotStructure]], structures))
            logging.info(f"Trained {match[AGENT_1]} vs {match[AGENT_2]} ({num_trainings+1}/{args.max_trainings})")
            num_trainings += 1
            save_evo_metadata(metadata_dir_path, num_trainings)

        # evaluate
        logging.info(f"# Start Evaluation (Generation {generation})")
        matches = get_matches(
            populations[AGENT_1].get_evaluation_indices(),
            populations[AGENT_2].get_evaluation_indices(),
            args.eval_num_opponents,
            metadata_dir_path,
            generation,
            "eval",
        )

        for match in matches:
            structures = {agent_id: populations[agent_id][id] for agent_id, id in match.items()}

            if structures[AGENT_1].has_fought(match[AGENT_2]):
                assert structures[AGENT_2].has_fought(match[AGENT_1])
                logging.info(f"Skipped evaluation {match[AGENT_1]} vs {match[AGENT_2]} (already fought)")
                continue

            results = evaluate(
                cast(Dict[AgentID, Union[Structure, DummyRobotStructure]], structures),
                args.env_name,
                num_processes=1,
                device=torch.device("cpu"),
                min_num_episodes=1,
                seed=None,
            )

            for agent_id in AGENT_IDS:
                opponent_id = get_opponent_id(agent_id)
                populations[agent_id].set_score(match[agent_id], match[opponent_id], results[agent_id])
            logging.info(
                f"Evaluated {match[AGENT_1]}({results[AGENT_1]:.3f}) " f"vs {match[AGENT_2]}({results[AGENT_2]:.3f})"
            )

        for name in AGENT_IDS:
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
        for name in AGENT_IDS:
            non_survivors = populations[name].update(num_survivors, num_reproductions)
            non_survivors_dict[name] = non_survivors
        for agent_id in AGENT_IDS:
            opponent_id = get_opponent_id(agent_id)
            for opponent_robot_id in non_survivors_dict[opponent_id]:
                populations[agent_id].delete_score(opponent_robot_id)

    logging.info("Experiment finished")
