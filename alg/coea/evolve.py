import argparse
import os
import random

from alg.coea.population import Population
from alg.ppo import train
from utils import get_agent_names


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.env_name, args.exp_dirname)
    os.makedirs(save_path)

    robot_ids = get_agent_names()

    populations = {id_: Population(os.path.join(save_path, id_), args) for id_ in robot_ids}

    while True:

        # pair robots
        indices = list(range(args.pop_size))
        random.shuffle(indices)

        for id1, id2 in zip(range(args.pop_size), indices):

            structures = {"robot_1": populations["robot_1"][id1], "robot_2": populations["robot_2"][id2]}
            train(args, structures)

        break
