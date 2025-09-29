import os
from pathlib import PurePath
from typing import List, Literal, Optional

import pandas as pd  # type: ignore

from utils import load_args


def get_env_name(experiment_dir: str) -> str:

    if extract_exp_type(experiment_dir) == "coea":
        coea_args = load_args(os.path.join(experiment_dir, "metadata"))
        env_name = coea_args.env_name
        return env_name
    else:
        raise NotImplementedError("Only coea experiments are supported.")


def extract_exp_type(experiment_dir: str) -> Literal["coea", "ppo"]:

    parts = PurePath(experiment_dir).parts

    if "experiments" not in parts:
        raise ValueError("The experiment directory path is not valid.")

    idx = parts.index("experiments")
    if idx + 1 >= len(parts):
        raise ValueError("The experiment directory path is not valid.")

    if parts[idx + 1] == "coea":
        return "coea"
    elif parts[idx + 1] == "ppo":
        return "ppo"
    else:
        raise ValueError("The experiment directory path is not valid.")


def get_top_robot_ids(csv_path: str, top_n: int = 1, generation: Optional[int] = None) -> List[int]:
    """
    Returns the indices of the top N robots based on their scores from a CSV file.
    Args:
        csv_path (str): Path to the CSV file containing robot scores.
                        The CSV should have a 'generation' column and columns for each robot.
        top_n (int, optional): Number of top robots to select. Defaults to 1.
        generation (Optional[int], optional): The generation to filter by.
                                              If None, uses the latest generation in the CSV. Defaults to None.
    Returns:
        List[int]: List of column indices corresponding to the top N robots in the specified generation.
    """

    df = pd.read_csv(csv_path)
    if generation is None:
        generation = df["generation"].max()
    df.drop(columns=["generation"], inplace=True)

    top_robots = df.loc[generation].nlargest(top_n).index.tolist()
    top_robot_indices = [df.columns.get_loc(col) for col in top_robots]

    return top_robot_indices


def get_robot_save_path(population_path: str, robot_id: int, generation: Optional[int] = None) -> str:
    """
    Returns the file path of the robot specified by robot_id and generation within a population directory.
    Args:
        population_path (str): The root directory path containing generations of robot data.
        robot_id (int): The unique identifier of the robot.
        generation (Optional[int], optional): The generation number to search in.
                                              If None, searches in the latest generation.
    Returns:
        str: The file path to the robot's directory.
    Raises:
        FileNotFoundError: If the robot's directory cannot be found in any generation.
    """

    if generation is None:
        generation = get_max_generation(population_path)

    while True:
        robot_save_path = os.path.join(population_path, f"generation{generation:02}", f"id{robot_id:02}")
        if os.path.exists(robot_save_path):
            return robot_save_path
        generation -= 1
        if generation < 0:
            raise FileNotFoundError(f"Robot {robot_id} not found in {population_path}")


def get_max_generation(population_path: str) -> int:
    """
    Returns the maximum generation number found in the specified directory.

    Args:
        population_path (str): The root directory path containing generations of robot data.
    Returns:
        int: The maximum generation number found.
    """

    generation = 0
    while os.path.exists(os.path.join(population_path, f"generation{generation+1:02}")):
        generation += 1
    return generation
