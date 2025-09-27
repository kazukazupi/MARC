import csv
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from evogym import hashable, sample_robot  # type: ignore

from alg.coea.structure import Structure, mutate
from utils import AgentID


class Population:

    def __init__(
        self,
        agent_id: AgentID,
        save_path: str,
        pop_size: int,
        robot_shape: Tuple[int, int],
        is_continuing: bool = False,
    ):

        self.agent_id = agent_id
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
            for robot_id in range(pop_size):

                body, connections = sample_robot(robot_shape)
                while hashable(body) in self.population_structure_hashes:
                    body, connections = sample_robot(robot_shape)

                self.structures.append(Structure(os.path.join(generation_path, f"id{robot_id:02}"), body, connections))
                self.population_structure_hashes[hashable(body)] = True

        else:
            assert os.path.exists(self.save_path)
            assert os.path.exists(self.csv_path)
            generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")

            while os.path.exists(generation_path):

                for robot_id in range(pop_size):
                    structure_path = os.path.join(generation_path, f"id{robot_id:02}")

                    if self.generation == 0:
                        assert os.path.exists(structure_path)
                        structure = Structure.from_save_path(structure_path)
                        self.structures.append(structure)
                    else:
                        if not os.path.exists(structure_path):
                            continue
                        structure = Structure.from_save_path(structure_path)
                        self.structures[robot_id] = structure

                    self.population_structure_hashes[hashable(structure.body)] = True

                self.generation += 1
                generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")

            self.generation -= 1

    def update(self, num_survivors: int, num_reproductions: int) -> List[int]:
        logging.info(f"## Updating {self.agent_id} population")

        # selection
        if any(fitness is None for fitness in self.fitnesses):
            raise ValueError("All fitnesses must be set before updating the population.")
        fitnesses_ = np.array(self.fitnesses)
        sorted_args = list(np.argsort(-fitnesses_))
        survivors = sorted_args[:num_survivors]
        non_survivors = sorted_args[num_survivors:]
        logging.info(f"Survivors: {','.join(map(str, survivors))}")
        for robot_id in non_survivors:
            self.structures[robot_id].is_died = True

        # reproduce
        self.generation += 1
        generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
        os.mkdir(generation_path)

        for child_robot_id in non_survivors[:num_reproductions]:
            child_save_path = os.path.join(generation_path, f"id{child_robot_id:02}")
            num_attempts = 100
            for _ in range(num_attempts):
                parent_robot_id = random.choice(survivors)
                child = mutate(self.structures[parent_robot_id], child_save_path, self.population_structure_hashes)
                if child is not None:
                    break
            else:
                raise RuntimeError("Failed to generate a child.")

            logging.info(f"Reproduced {parent_robot_id} -> {child_robot_id}")
            self.structures[child_robot_id] = child
            self.population_structure_hashes[hashable(child.body)] = True

        return non_survivors

    def get_training_indices(self) -> List[int]:
        indices = [idx for idx, structure in enumerate(self.structures) if not structure.is_trained]
        return indices

    def get_evaluation_indices(self) -> List[int]:
        indices = [
            idx for idx, structure in enumerate(self.structures) if structure.is_trained and not structure.is_died
        ]
        return indices

    def set_score(self, self_robot_id: int, opponent_robot_id: int, score: float) -> None:
        self.structures[self_robot_id].set_score(opponent_robot_id, score)

    def delete_score(self, opponent_robot_id: int) -> None:
        for structure in self.structures:
            if structure.has_fought(opponent_robot_id):
                structure.delete_score(opponent_robot_id)

    @property
    def fitnesses(self) -> List[Optional[float]]:
        return [structure.fitness for structure in self.structures]

    def dump_fitnesses(self) -> None:
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.generation] + self.fitnesses)

    def __getitem__(self, index: int) -> Structure:
        return self.structures[index]
