import argparse
import os
import random
from typing import Dict, List

import numpy as np
from evogym import hashable, sample_robot  # type: ignore

from alg.coea.structure import Structure, mutate


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

    def update(self, num_survivors: int) -> None:

        # selection
        sorted_args = np.argsort(-self.fitnesses)
        survivors = sorted_args[:num_survivors]
        non_survivors = sorted_args[num_survivors:]

        # reproduce
        self.generation += 1
        generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
        os.mkdir(generation_path)

        for id_ in non_survivors:
            child_save_path = os.path.join(generation_path, f"id{id_:02}")
            num_attempts = 100
            for _ in range(num_attempts):
                parent_id = random.choice(survivors)
                child = mutate(self.structures[parent_id], child_save_path, self.population_structure_hashes)
                if child is not None:
                    break
            else:
                raise RuntimeError("Failed to generate a child.")

            self.structures[id_] = child
            self.population_structure_hashes[hashable(child.body)] = True

    def get_training_indices(self) -> List[int]:
        indices = [idx for idx, structure in enumerate(self.structures) if not structure.is_trained]
        return indices

    def get_evaluation_indices(self) -> List[int]:
        indices = [idx for idx, structure in enumerate(self.structures) if structure.is_trained]
        return indices

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([structure.fitness for structure in self.structures])

    @fitnesses.setter
    def fitnesses(self, fitnesses: np.ndarray) -> None:
        if len(fitnesses) != len(self.structures):
            raise ValueError("Length of fitnesses does not match the number of structures.")
        for structure, fitness in zip(self.structures, fitnesses):
            structure.fitness = fitness

    def __getitem__(self, index: int) -> Structure:
        return self.structures[index]
