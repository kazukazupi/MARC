import csv
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from evogym import hashable, sample_robot  # type: ignore

from alg.coea.structure import Structure, mutate


class Population:

    def __init__(
        self, agent_name: str, save_path: str, pop_size: int, robot_shape: Tuple[int, int], is_continuing: bool = False
    ):

        self.agent_name = agent_name
        self.save_path = save_path
        self.csv_path = os.path.join(self.save_path, "fitnesses.csv")
        self.structures: List[Structure] = []
        self.population_structure_hashes: Dict[str, bool] = {}
        self.generation = 0

        if not is_continuing:

            # create log files
            os.mkdir(self.save_path)
            with open(self.csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["generation"] + [f"id{i:02}" for i in range(pop_size)])
            generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
            os.mkdir(generation_path)

            # generate a population
            for id_ in range(pop_size):

                body, connections = sample_robot(robot_shape)
                while hashable(body) in self.population_structure_hashes:
                    body, connections = sample_robot(robot_shape)

                self.structures.append(Structure(os.path.join(generation_path, f"id{id_:02}"), body, connections))
                self.population_structure_hashes[hashable(body)] = True

        else:
            assert os.path.exists(self.save_path)
            assert os.path.exists(self.csv_path)
            generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")

            while os.path.exists(generation_path):

                for id_ in range(pop_size):
                    structure_path = os.path.join(generation_path, f"id{id_:02}")

                    if self.generation == 0:
                        assert os.path.exists(structure_path)
                        structure = Structure.from_save_path(structure_path)
                        self.structures.append(structure)
                    else:
                        if not os.path.exists(structure_path):
                            continue
                        structure = Structure.from_save_path(structure_path)
                        self.structures[id_] = structure

                    self.population_structure_hashes[hashable(structure.body)] = True

                self.generation += 1
                generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")

            self.generation -= 1

    def update(self, num_survivors: int, num_reproductions: int):
        logging.info(f"Updating {self.agent_name} population")

        # selection
        if any(fitness is None for fitness in self.fitnesses):
            raise ValueError("All fitnesses must be set before updating the population.")
        fitnesses_ = np.array(self.fitnesses)
        sorted_args = np.argsort(-fitnesses_)
        survivors = sorted_args[:num_survivors]
        non_survivors = sorted_args[num_survivors:]
        logging.info(f"Survivors: {','.join(map(str, survivors))}")
        for id_ in non_survivors:
            self.structures[id_].is_died = True

        # reproduce
        self.generation += 1
        generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
        os.mkdir(generation_path)

        for id_ in non_survivors[:num_reproductions]:
            child_save_path = os.path.join(generation_path, f"id{id_:02}")
            num_attempts = 100
            for _ in range(num_attempts):
                parent_id = random.choice(survivors)
                child = mutate(self.structures[parent_id], child_save_path, self.population_structure_hashes)
                if child is not None:
                    break
            else:
                raise RuntimeError("Failed to generate a child.")

            logging.info(f"Reproduced {parent_id} -> {id_}")
            self.structures[id_] = child
            self.population_structure_hashes[hashable(child.body)] = True

    def get_training_indices(self) -> List[int]:
        indices = [idx for idx, structure in enumerate(self.structures) if not structure.is_trained]
        return indices

    def get_evaluation_indices(self) -> List[int]:
        indices = [
            idx for idx, structure in enumerate(self.structures) if structure.is_trained and not structure.is_died
        ]
        return indices

    @property
    def fitnesses(self) -> List[Optional[float]]:
        return [structure.fitness for structure in self.structures]

    @fitnesses.setter
    def fitnesses(self, fitnesses: List[Optional[float]]) -> None:
        if len(fitnesses) != len(self.structures):
            raise ValueError("Length of fitnesses does not match the number of structures.")

        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.generation] + list(fitnesses))

        for structure, fitness in zip(self.structures, fitnesses):
            structure.fitness = fitness

    def __getitem__(self, index: int) -> Structure:
        return self.structures[index]
