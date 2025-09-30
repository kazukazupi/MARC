import argparse
import json
import os
import warnings

import torch

from utils.agent_id import AGENT_1, AGENT_2


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--env-name", type=str, default="Sumo-v0", help="environment name")
    parser.add_argument("--exp-dirname", type=str, default="log", help="directory name to save models")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--is-continue", action="store_true", help="continue training from a saved model")

    # Dummy Robot Specification
    parser.add_argument(
        "--dummy-target", choices=[AGENT_1, AGENT_2], default=AGENT_2, help="which agent is the dummy robot"
    )
    parser.add_argument(
        "--dummy-body-type",
        type=str,
        choices=["rigid_4x4", "soft_4x4", "rigid_5x5", "soft_5x5"],
        default="soft_4x4",
        help="type of dummy robot",
    )

    # Coevolution Hyperparameters
    parser.add_argument("--pop-size", type=int, default=25, help="population size")
    parser.add_argument("--max-trainings", type=int, default=250, help="maximum number of trainings")
    parser.add_argument("--robot-shape", type=tuple, default=(5, 5), help="shape of the robot")
    parser.add_argument("--eval-num-opponents", type=int, default=25, help="number of opponents to evaluate")

    # PPO Logging
    parser.add_argument("--log-interval", type=int, default=1, help="interval between logging")
    parser.add_argument("--eval-interval", type=int, default=10, help="interval between evaluations")
    parser.add_argument("--save-interval", type=int, default=10, help="interval between saving models")
    parser.add_argument("--num-evals", type=int, default=1, help="number of evaluations")

    # PPO Hyperparameters
    parser.add_argument("--delta", type=float, default=0.0, help="delta for self-play opponent sampling")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
    parser.add_argument("--num-updates", type=int, default=1000, help="number of updates")
    parser.add_argument("--num-processes", type=int, default=1, help="number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128, help="number of forward steps in A2C")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="gae lambda parameter")
    parser.add_argument("--clip-param", type=float, default=0.1, help="ppo clip parameter")
    parser.add_argument("--ppo-epoch", type=int, default=4, help="number of ppo epochs")
    parser.add_argument("--num-mini-batch", type=int, default=4, help="number of ppo mini batches")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max norm of gradients")
    parser.add_argument("--use-gae", type=bool, default=True, help="use generalized advantage estimation")
    parser.add_argument(
        "--use-proper-time-limits", type=bool, default=False, help="compute returns taking into account time limits"
    )

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.pop_size < args.eval_num_opponents:
        warnings.warn("eval_num_opponents should be less than or equal to pop_size")

    return args


def save_args(args: argparse.Namespace, metadata_dir_path: str) -> None:
    save_path = os.path.join(metadata_dir_path, "args.json")
    with open(save_path, "w") as f:
        args_dict = vars(args).copy()
        args_dict.pop("device", None)
        json.dump(args_dict, f, indent=4)


def load_args(metadata_dir_path: str) -> argparse.Namespace:
    save_path = os.path.join(metadata_dir_path, "args.json")
    with open(save_path, "r") as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args
