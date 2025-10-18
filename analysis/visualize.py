import argparse
import json
import os
from typing import Dict, Optional

import torch

from alg.coea.structure import BaseRobotStructure, DummyRobotStructure, Structure
from alg.ppo import evaluate
from analysis.analysis_utils import extract_exp_type, get_env_name, get_robot_save_path, get_top_robot_ids
from utils import AGENT_IDS, AgentID, get_opponent_id, load_args


def load_structure_coea(
    experiment_dir: str,
    agent_id: AgentID,
    generation: Optional[int] = None,
    id: Optional[int] = None,
) -> BaseRobotStructure:

    csv_path = os.path.join(experiment_dir, agent_id, "fitnesses.csv")
    if id is None:
        id = get_top_robot_ids(csv_path, generation=generation)[0]
    save_path = get_robot_save_path(os.path.join(experiment_dir, agent_id), id, generation)
    print(f"Loading {save_path}")
    return Structure.from_save_path(save_path)


def load_structure_ppo(experiment_dir: str, agent_id: AgentID) -> BaseRobotStructure:

    save_path = os.path.join(experiment_dir, agent_id)
    print(f"Loading {save_path}")
    return Structure.from_save_path(save_path)


def load_structure_coea_single(
    experiment_dir: str,
    agent_id: AgentID,
    generation: Optional[int] = None,
    id: Optional[int] = None,
) -> BaseRobotStructure:

    # Load args from metadata
    metadata_dir_path = os.path.join(experiment_dir, "metadata")
    args = load_args(metadata_dir_path)

    # Determine which agent is evolved
    opponent_robot_id: AgentID = args.dummy_target
    self_robot_id: AgentID = get_opponent_id(opponent_robot_id)

    if agent_id == self_robot_id:
        # Load evolved robot
        csv_path = os.path.join(experiment_dir, self_robot_id, "fitnesses.csv")
        if id is None:
            id = get_top_robot_ids(csv_path, generation=generation)[0]
        save_path = get_robot_save_path(os.path.join(experiment_dir, self_robot_id), id, generation)
        print(f"Loading {save_path}")
        return Structure.from_save_path(save_path)
    else:
        # Return dummy robot
        print(f"Loading dummy robot ({args.dummy_body_type})")
        return DummyRobotStructure(body_type=args.dummy_body_type)


def load_structure_ppo_single(experiment_dir: str, agent_id: AgentID) -> BaseRobotStructure:

    with open(os.path.join(experiment_dir, "env_info.json"), "r") as f:
        env_info = json.load(f)

    self_robot_id = env_info["self_robot_id"]
    dummy_body_type = env_info["dummy_body_type"]

    if agent_id == self_robot_id:
        # Load trained robot
        save_path = os.path.join(experiment_dir, self_robot_id)
        print(f"Loading {save_path}")
        return Structure.from_save_path(save_path)
    else:
        # Return dummy robot
        print(f"Loading dummy robot ({dummy_body_type})")
        return DummyRobotStructure(body_type=dummy_body_type)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to experiment directories. Provide 1 or 2 paths. If 1 path, it will be used for both agents.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        help="List of generations to visualize (1 or 2 values). If not provided, the latest generation will be used.",
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        help="List of robot ids to visualize (1 or 2 values). If not provided, the best robot id will be used.",
    )
    parser.add_argument(
        "--movie-path",
        type=str,
        default=None,
        help="Path to save the movie. If not provided, it will be shown on screen.",
    )
    parser.add_argument("--disable-tracking", action="store_true", help="Disable tracking")

    args = parser.parse_args()

    # Process experiment directories
    if len(args.experiment_dirs) == 1:
        experiment_dirs = args.experiment_dirs * 2
    elif len(args.experiment_dirs) == 2:
        experiment_dirs = args.experiment_dirs
    else:
        raise ValueError("Provide 1 or 2 experiment directories.")

    # Process generations
    if args.generations is None:
        generations = [None, None]
    elif len(args.generations) == 1:
        generations = args.generations * 2
    elif len(args.generations) == 2:
        generations = args.generations
    else:
        raise ValueError("Provide 1 or 2 generation values.")

    # Process ids
    if args.id is None:
        ids = [None, None]
    elif len(args.id) == 1:
        ids = args.id * 2
    elif len(args.id) == 2:
        ids = args.id
    else:
        raise ValueError("Provide 1 or 2 id values.")

    # Load structures for each agent
    structures: Dict[AgentID, BaseRobotStructure] = {}

    for i, (agent_id, experiment_dir) in enumerate(zip(AGENT_IDS, experiment_dirs)):
        exp_type = extract_exp_type(experiment_dir)
        generation = generations[i]
        id = ids[i]

        if exp_type == "coea":
            structures[agent_id] = load_structure_coea(experiment_dir, agent_id, generation, id)
        elif exp_type == "coea_single":
            structures[agent_id] = load_structure_coea_single(experiment_dir, agent_id, generation, id)
        elif exp_type == "ppo":
            structures[agent_id] = load_structure_ppo(experiment_dir, agent_id)
        elif exp_type == "ppo_single":
            structures[agent_id] = load_structure_ppo_single(experiment_dir, agent_id)
        else:
            raise NotImplementedError(f"Experiment type '{exp_type}' is not supported.")

    # Get environment name from first experiment directory
    env_name = get_env_name(experiment_dirs[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.movie_path is not None:
        os.makedirs(os.path.dirname(args.movie_path), exist_ok=True)

    # visualize
    evaluate(
        structures,
        env_name,
        num_processes=1,
        device=device,
        min_num_episodes=1,
        render_mode="human" if args.movie_path is None else "rgb_array",
        render_options={"disable_tracking": True} if args.disable_tracking else None,
        movie_path=args.movie_path,
    )
