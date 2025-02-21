import argparse
import os

from alg.coea.population import Population


def evolve(args: argparse.Namespace):

    save_path = os.path.join("experiments", "coea", args.exp_dirname)
    os.makedirs(save_path)
    population = Population(args)
