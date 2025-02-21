import argparse
from typing import Dict, List

from evogym import hashable, sample_robot  # type: ignore

from alg.coea.structure import Structure


class Population:

    def __init__(self, args: argparse.Namespace):

        self.structures: List[Structure] = []
        self.population_structure_hashes: Dict[str, bool] = {}

        # generate a population
        for i in range(args.pop_size):

            body, connections = sample_robot(args.robot_shape)
            while hashable(body) in self.population_structure_hashes:
                body, connections = sample_robot(args.robot_shape)

            self.structures.append(Structure(body, connections))
            self.population_structure_hashes[hashable(body)] = True
