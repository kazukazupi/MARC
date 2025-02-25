import glob
import json
import os
from typing import Dict, Optional

import numpy as np
from evogym import draw, get_full_connectivity, get_uniform, has_actuator, hashable, is_connected  # type: ignore


class Structure:

    def __init__(self, save_path: str, body: np.ndarray, connections: np.ndarray, save: bool = True):

        self.save_path = save_path
        self.body = body
        self.connections = connections
        self.fitness = -np.inf

        if save:
            os.mkdir(self.save_path)
            np.save(os.path.join(self.save_path, "body.npy"), body)
            np.save(os.path.join(self.save_path, "connections.npy"), connections)
            with open(os.path.join(self.save_path, "metadata.json"), "w") as f:
                metadata = {"is_trained": False, "is_died": False}
                json.dump(metadata, f, indent=4)

    def get_latest_controller_path(self) -> str:
        controller_paths = sorted(glob.glob(os.path.join(self.save_path, "controller_*.pt")))
        assert controller_paths, f"Controller for {self.save_path} is not found."
        return max(controller_paths, key=os.path.getctime)

    @classmethod
    def from_save_path(cls, save_path: str) -> "Structure":
        body = np.load(os.path.join(save_path, "body.npy"))
        connections = np.load(os.path.join(save_path, "connections.npy"))
        return cls(save_path, body, connections, save=False)

    @property
    def is_trained(self) -> bool:
        with open(os.path.join(self.save_path, "metadata.json")) as f:
            metadata = json.load(f)
        return metadata["is_trained"]

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        with open(os.path.join(self.save_path, "metadata.json"), "r+") as f:
            metadata = json.load(f)
            metadata["is_trained"] = value
            f.seek(0)
            json.dump(metadata, f, indent=4)
            f.truncate()

    @property
    def is_died(self) -> bool:
        with open(os.path.join(self.save_path, "metadata.json")) as f:
            metadata = json.load(f)
        return metadata["is_died"]

    @is_died.setter
    def is_died(self, value: bool) -> None:
        with open(os.path.join(self.save_path, "metadata.json"), "r+") as f:
            metadata = json.load(f)
            metadata["is_died"] = value
            f.seek(0)
            json.dump(metadata, f, indent=4)
            f.truncate()


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
