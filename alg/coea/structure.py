import glob
import os

import numpy as np


class Structure:

    def __init__(self, save_path: str, body: np.ndarray, connections: np.ndarray):

        self.save_path = save_path
        self.body = body
        self.connections = connections

        os.mkdir(self.save_path)
        np.save(os.path.join(self.save_path, "body.npy"), body)
        np.save(os.path.join(self.save_path, "connections.npy"), connections)

    def get_latest_controller_path(self) -> str:
        controller_paths = sorted(glob.glob(os.path.join(self.save_path, "controller_*.pt")))
        assert controller_paths, f"Controller for {self.save_path} is not found."
        return max(controller_paths, key=os.path.getctime)
