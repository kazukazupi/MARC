import argparse
import logging
import math
import os
from typing import Dict, Union, cast

import torch

from alg.coea.coea_utils import get_percent_survival_evals, load_evo_metadata, save_evo_metadata
from alg.coea.population import Population
from alg.coea.structure import DummyRobotStructure, Structure
from alg.evaluate import evaluate
from alg.ppo import train
from utils import AgentID, get_opponent_id, load_args, save_args


def evolve_single(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea_single", args.env_name, args.exp_dirname)
    metadata_dir_path = os.path.join(save_path, "metadata")
    log_file = os.path.join(save_path, "experiment.log")

    # Determine agent IDs
    opponent_robot_id: AgentID = args.dummy_target
    self_robot_id: AgentID = get_opponent_id(opponent_robot_id)

    if not args.is_continue:

        os.makedirs(save_path)
        os.mkdir(metadata_dir_path)

        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s",
        )
        logging.info(f"Starting single agent evolution at {save_path}")
        logging.info(f"Self agent: {self_robot_id}, Opponent (dummy): {opponent_robot_id}")

        save_args(args, metadata_dir_path)
        save_evo_metadata(metadata_dir_path, 0)

        population = Population(self_robot_id, os.path.join(save_path, self_robot_id), args.pop_size, args.robot_shape)
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
        logging.info(f"Continuing single agent evolution at {save_path}")

        args = load_args(metadata_dir_path)
        num_trainings = load_evo_metadata(metadata_dir_path)
        opponent_robot_id = args.dummy_target
        self_robot_id = get_opponent_id(opponent_robot_id)
        population = Population(
            self_robot_id, os.path.join(save_path, self_robot_id), args.pop_size, args.robot_shape, is_continuing=True
        )

    # Create dummy opponent structure
    opponent_structure = DummyRobotStructure(args.dummy_body_type)

    while True:
        generation = population.generation
        logging.info(f"# Start Training (Generation {generation})")

        # train
        training_indices = population.get_training_indices()

        for robot_id in training_indices:
            if num_trainings >= args.max_trainings:
                break

            structures = {
                self_robot_id: population[robot_id],
                opponent_robot_id: opponent_structure,
            }
            train(args, cast(Dict[AgentID, Union[Structure, DummyRobotStructure]], structures))
            logging.info(f"Trained {robot_id} ({num_trainings+1}/{args.max_trainings})")
            num_trainings += 1
            save_evo_metadata(metadata_dir_path, num_trainings)

        # evaluate
        logging.info(f"# Start Evaluation (Generation {generation})")
        evaluation_indices = population.get_evaluation_indices()

        for robot_id in evaluation_indices:
            structures = {
                self_robot_id: population[robot_id],
                opponent_robot_id: opponent_structure,
            }

            # Since we're fighting against a dummy, we use a simple opponent_id for tracking
            # We use -1 to indicate dummy opponent
            dummy_opponent_id = -1

            if population[robot_id].has_fought(dummy_opponent_id):
                logging.info(f"Skipped evaluation {robot_id} (already fought dummy)")
                continue

            results = evaluate(
                cast(Dict[AgentID, Union[Structure, DummyRobotStructure]], structures),
                args.env_name,
                num_processes=1,
                device=torch.device("cpu"),
                min_num_episodes=1,
                seed=None,
            )

            population.set_score(robot_id, dummy_opponent_id, results[self_robot_id])
            logging.info(f"Evaluated {robot_id} ({results[self_robot_id]:.3f})")

        population.dump_fitnesses()

        if num_trainings >= args.max_trainings:
            break

        # selection, reproduction
        logging.info(f"# Start Selection, Reproduction (Generation {generation})")
        percent_survival = get_percent_survival_evals(num_trainings, args.max_trainings)
        num_survivors = max(2, math.ceil(args.pop_size * percent_survival))
        num_reproductions = min(args.pop_size - num_survivors, args.max_trainings - num_trainings)
        logging.info(f"Percent survival: {percent_survival * 100:.2f}%")
        logging.info(f"Num survivors: {num_survivors}")
        logging.info(f"Num reproductions: {num_reproductions}")
        population.update(num_survivors, num_reproductions)

    logging.info("Single agent evolution finished")
