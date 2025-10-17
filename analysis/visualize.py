import argparse
import json
import os
from typing import Dict, List, Optional, Union

import torch

from alg.coea.structure import DummyRobotStructure, Structure
from alg.ppo import evaluate
from analysis.analysis_utils import extract_exp_type, get_env_name, get_robot_save_path, get_top_robot_ids
from utils import AGENT_IDS, AgentID, get_opponent_id


def load_structures_coea(
    experiment_dir: str,
    generations: Optional[List[Optional[int]]] = None,
    ids: Optional[List[int]] = None,
):

    # Process generations, ids
    if generations is None:
        generations = [None] * 2
    elif len(generations) == 1:
        generations = generations * 2
    elif len(generations) == 2:
        pass
    else:
        raise ValueError("Invalid number of generations.")

    if ids is not None:
        assert len(ids) == 2, "The number of ids must match the number of agents."

    # Load structures
    structures: Dict[AgentID, Structure] = {}
    for i, (a, generation) in enumerate(zip(AGENT_IDS, generations)):
        csv_path = os.path.join(experiment_dir, a, "fitnesses.csv")
        if ids is None:
            id_ = get_top_robot_ids(csv_path, generation=generation)[0]
        else:
            id_ = ids[i]
        save_path = get_robot_save_path(os.path.join(experiment_dir, a), id_, generation)
        print(f"Loading {save_path}")
        structures[a] = Structure.from_save_path(save_path)

    return structures


def load_structures_ppo(experiment_dir: str):

    structures: Dict[AgentID, Structure] = {
        a: Structure.from_save_path(os.path.join(experiment_dir, a)) for a in AGENT_IDS
    }

    return structures


def load_structures_ppo_single(experiment_dir: str):

    with open(os.path.join(experiment_dir, "env_info.json"), "r") as f:
        env_info = json.load(f)

    self_robot_id = env_info["self_robot_id"]
    opponent_robot_id = get_opponent_id(self_robot_id)
    dummy_body_type = env_info["dummy_body_type"]

    structures: Dict[AgentID, Union[Structure, DummyRobotStructure]] = {
        self_robot_id: Structure.from_save_path(os.path.join(experiment_dir, self_robot_id)),
        opponent_robot_id: DummyRobotStructure(body_type=dummy_body_type),
    }

    return structures


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        help="List of generations to visualize. If not provided, the latest generation will be used.",
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        help="List of robot ids to visualize. If not provided, the top robot id will be used.",
    )
    parser.add_argument(
        "--movie-path",
        type=str,
        default=None,
        help="Path to save the movie. If not provided, it will be shown on screen.",
    )
    parser.add_argument("--disable-tracking", action="store_true", help="Disable tracking")

    args = parser.parse_args()

    # load structures
    exp_type = extract_exp_type(args.experiment_dir)
    structures: Dict[AgentID, Union[Structure, DummyRobotStructure]]
    if exp_type == "coea":
        structures = load_structures_coea(
            args.experiment_dir,
            generations=args.generations,
            ids=args.id,
        )
    elif exp_type == "ppo":
        structures = load_structures_ppo(args.experiment_dir)
    elif exp_type == "ppo_single":
        structures = load_structures_ppo_single(args.experiment_dir)
    else:
        raise NotImplementedError("Only coea experiments are supported.")

    env_name = get_env_name(args.experiment_dir)
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
