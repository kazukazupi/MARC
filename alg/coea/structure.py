import numpy as np


class Structure:

    def __init__(self, save_path: str, body: np.ndarray, connections: np.ndarray):
        self.save_path = save_path
        self.body = body
        self.connections = connections
