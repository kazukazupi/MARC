import argparse
import os
from typing import Dict, List

import numpy as np
from evogym import hashable, sample_robot  # type: ignore

from alg.coea.structure import Structure


class Population:

    def __init__(self, save_path: str, args: argparse.Namespace):

        self.save_path = save_path
        os.mkdir(self.save_path)

        self.structures: List[Structure] = []
        self.population_structure_hashes: Dict[str, bool] = {}

        self.generation = 0
        generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
        os.mkdir(generation_path)

        # generate a population
        for id_ in range(args.pop_size):

            body, connections = sample_robot(args.robot_shape)
            while hashable(body) in self.population_structure_hashes:
                body, connections = sample_robot(args.robot_shape)

            self.structures.append(Structure(os.path.join(generation_path, f"id{id_:02}"), body, connections))
            self.population_structure_hashes[hashable(body)] = True

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([structure.fitness for structure in self.structures])

    @fitnesses.setter
    def fitnesses(self, fitnesses: np.ndarray) -> None:
        for structure, fitness in zip(self.structures, fitnesses):
            structure.fitness = fitness

    def __getitem__(self, index: int) -> Structure:
        return self.structures[index]
