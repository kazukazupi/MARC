import numpy as np


class Structure:

    def __init__(self, body: np.ndarray, connections: np.ndarray):
        self.body = body
        self.connections = connections
