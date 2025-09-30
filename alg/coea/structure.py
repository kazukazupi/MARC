import glob
import json
import os
from typing import Dict, Literal, Optional

import numpy as np
from evogym import draw, get_full_connectivity, get_uniform, has_actuator, hashable, is_connected  # type: ignore

from alg.coea.coea_utils import StructureMetadata


class Structure:

    def __init__(self, save_path: str, body: np.ndarray, connections: np.ndarray, save: bool = True):

        self.save_path = save_path
        self.body = body
        self.connections = connections

        if save:
            os.mkdir(self.save_path)
            np.save(os.path.join(self.save_path, "body.npy"), body)
            np.save(os.path.join(self.save_path, "connections.npy"), connections)
            self.metadata = StructureMetadata(is_trained=False, is_died=False)
            self.dump_metadata()
        else:
            with open(os.path.join(self.save_path, "metadata.json"), "r") as f:
                metadata_dict = json.load(f)
            self.metadata = StructureMetadata(**metadata_dict)

    def get_latest_controller_path(self) -> str:
        controller_paths = sorted(glob.glob(os.path.join(self.save_path, "controller_*.pt")))
        assert controller_paths, f"Controller for {self.save_path} is not found."
        return max(controller_paths, key=os.path.getctime)

    @classmethod
    def from_save_path(cls, save_path: str) -> "Structure":
        body = np.load(os.path.join(save_path, "body.npy"))
        connections = np.load(os.path.join(save_path, "connections.npy"))
        return cls(save_path, body, connections, save=False)

    def has_fought(self, opponent_id: int) -> bool:
        return opponent_id in self.metadata.scores

    def set_score(self, opponent_id: int, score: float) -> None:
        self.metadata.scores[opponent_id] = score
        self.dump_metadata()

    def delete_score(self, opponent_id: int) -> None:
        del self.metadata.scores[opponent_id]
        self.dump_metadata()

    @property
    def fitness(self) -> Optional[float]:
        if self.is_died:
            return None
        if not self.metadata.scores:
            return None
        values = list(self.metadata.scores.values())
        return float(np.mean(values))

    @property
    def is_trained(self) -> bool:
        return self.metadata.is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self.metadata.is_trained = value
        self.dump_metadata()

    @property
    def is_died(self) -> bool:
        return self.metadata.is_died

    @is_died.setter
    def is_died(self, value: bool) -> None:
        self.metadata.is_died = value
        self.dump_metadata()

    def dump_metadata(self) -> None:
        with open(os.path.join(self.save_path, "metadata.json"), "w") as f:
            json.dump(self.metadata.model_dump(), f, indent=4)


def mutate(
    structure: Structure,
    child_save_path: str,
    population_structure_hashes: Dict[str, bool],
    mutation_rate: float = 0.1,
    num_attempts: int = 10,
) -> Optional[Structure]:

    body = structure.body.copy()

    pd = get_uniform(5)
    pd[0] = 0.6

    for n in range(num_attempts):
        for i in range(body.shape[0]):
            for j in range(body.shape[1]):
                mutation = [mutation_rate, 1 - mutation_rate]
                if draw(mutation) == 0:
                    body[i][j] = draw(pd)

        if is_connected(body) and has_actuator(body) and hashable(body) not in population_structure_hashes:
            connections = get_full_connectivity(body)
            return Structure(child_save_path, body, connections)

    return None


class DummyRobotStructure:

    def __init__(self, body_type: Literal["rigid_4x4", "soft_4x4", "rigid_5x5", "soft_5x5"]):

        self.body_type = body_type

        if body_type == "rigid_4x4":
            self.body = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                ]
            )
        elif body_type == "soft_4x4":
            self.body = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 0],
                    [2, 2, 2, 2, 0],
                    [2, 2, 2, 2, 0],
                    [2, 2, 2, 2, 0],
                ]
            )
        elif body_type == "rigid_5x5":
            self.body = np.ones((5, 5))
        elif body_type == "soft_5x5":
            self.body = np.full((5, 5), 2)
        else:
            raise ValueError(f"Invalid body_type: {body_type}")

        self.connections = get_full_connectivity(self.body)
