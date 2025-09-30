from typing import Literal, Tuple

import numpy as np
from evogym import get_full_connectivity  # type: ignore


def get_dummy_robot(mode: Literal["rigid_4x4", "soft_4x4", "rigid_5x5", "soft_5x5"]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a dummy robot body and its connections based on the specified mode.
    Modes:
    - "rigid_4x4": 4x4 rigid robot
    - "soft_4x4": 4x4 soft robot
    - "rigid_5x5": 5x5 rigid robot
    - "soft_5x5": 5x5 soft robot
    Returns:
        body (np.ndarray): 2D array representing the robot's body.
        connections (np.ndarray): 2D array representing the robot's connections.
    """

    if mode == "rigid_4x4":
        body = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ]
        )
    elif mode == "soft_4x4":
        body = np.array(
            [
                [0, 0, 0, 0, 0],
                [2, 2, 2, 2, 0],
                [2, 2, 2, 2, 0],
                [2, 2, 2, 2, 0],
                [2, 2, 2, 2, 0],
            ]
        )
    elif mode == "rigid_5x5":
        body = np.ones((5, 5))
    elif mode == "soft_5x5":
        body = np.full((5, 5), 2)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    connections = get_full_connectivity(body)

    return body, connections
