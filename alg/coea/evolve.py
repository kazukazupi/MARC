import argparse
import os

from alg.coea.population import Population


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.exp_dirname)
    os.makedirs(save_path)

    robot_ids = ["robot_1", "robot_2"]

    populations = {id_: Population(os.path.join(save_path, id_), args) for id_ in robot_ids}
