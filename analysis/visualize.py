import argparse
import os
from typing import Dict

import torch

from alg.coea.structure import Structure
from alg.ppo import evaluate
from analysis.analysis_utils import get_env_name, get_robot_save_path, get_top_robot_ids
from utils import AGENT_IDS, AgentID

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

    # Process generations, ids
    if args.generations is None:
        args.generations = [None] * 2
    elif len(args.generations) == 1:
        args.generations = args.generations * 2
    elif len(args.generations) == 2:
        pass
    else:
        raise ValueError("Invalid number of generations.")

    if args.id is not None:
        assert len(args.id) == 2, "The number of ids must match the number of agents."

    # Load structures
    structures: Dict[AgentID, Structure] = {}
    for i, (a, generation) in enumerate(zip(AGENT_IDS, args.generations)):
        csv_path = os.path.join(args.experiment_dir, a, "fitnesses.csv")
        if args.id is None:
            id_ = get_top_robot_ids(csv_path, generation=generation)[0]
        else:
            id_ = args.id[i]
        save_path = get_robot_save_path(os.path.join(args.experiment_dir, a), id_, generation)
        print(f"Loading {save_path}")
        structures[a] = Structure.from_save_path(save_path)

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
