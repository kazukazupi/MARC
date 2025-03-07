import os
from typing import List, Optional

import pandas as pd  # type: ignore


def get_top_robot_ids(csv_path: str, top_n: int = 1, generation: Optional[int] = None) -> List[int]:
    df = pd.read_csv(csv_path)
    if generation is None:
        generation = df["generation"].max()
    df.drop(columns=["generation"], inplace=True)

    top_robots = df.loc[generation].nlargest(top_n).index.tolist()
    top_robot_indices = [df.columns.get_loc(col) for col in top_robots]

    return top_robot_indices


def get_robot_save_path(population_path: str, robot_id: int, generation: Optional[int] = None) -> str:
    if generation is None:
        generation = get_max_generation(population_path)

    while True:
        robot_save_path = os.path.join(population_path, f"generation{generation:02}", f"id{robot_id:02}")
        if os.path.exists(robot_save_path):
            return robot_save_path
        generation -= 1
        if generation < 0:
            raise FileNotFoundError(f"Robot {robot_id} not found in {population_path}")


def get_max_generation(path: str):
    generation = 0
    while os.path.exists(os.path.join(path, f"generation{generation+1:02}")):
        generation += 1
    return generation
