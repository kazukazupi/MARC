import argparse
import os

import cv2
import numpy as np

from analysis.analysis_utils import *


def color(val: int) -> tuple[int, int, int]:
    """
    Returns the color associated with a voxel type.

    Args:
        val (int): The voxel type value.
    Returns:
        tuple[int, int, int]: The RGB color associated with the voxel type.
    """

    if val == 0:
        return (255, 255, 255)
    elif val == 1:
        return (34, 34, 34)
    elif val == 2:
        return (183, 183, 183)
    elif val == 3:
        return (66, 124, 252)
    elif val == 4:
        return (205, 167, 99)
    else:
        raise ValueError(f"Unknown body part value: {val}")


def get_image(body: np.ndarray) -> np.ndarray:
    """
    Returns an image representation of the body.

    Args:
        body (np.ndarray): The body part voxel grid.

    Returns:
        np.ndarray: The image representation of the body.
    """

    H, W = body.shape

    img = np.full((50 * H, 50 * W, 3), 255, dtype=np.uint8)

    for h in range(H):
        for w in range(W):
            cv2.rectangle(img, (w * 50, h * 50), ((w + 1) * 50, (h + 1) * 50), color(body[h][w]), thickness=-1)

    return img


def get_best_robot_images(experiment_dir: str, generations: Optional[List[int]] = None, top_n: int = 4):

    # read csv
    population_path_1 = os.path.join(experiment_dir, "robot_1")
    population_path_2 = os.path.join(experiment_dir, "robot_2")

    csv_path_1 = os.path.join(population_path_1, "fitnesses.csv")
    csv_path_2 = os.path.join(population_path_2, "fitnesses.csv")

    if generations is None:
        generations = [get_max_generation(population_path_1)]

    # make output dir
    env_name = get_env_name(experiment_dir)
    basename = os.path.basename(experiment_dir)
    output_dir = os.path.join("analysis", "figures", "body_image", env_name, basename)
    output_dir_1 = os.path.join(output_dir, "robot_1")
    output_dir_2 = os.path.join(output_dir, "robot_2")
    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)

    for generation in generations:

        robot_id_list_1 = get_top_robot_ids(csv_path_1, top_n=top_n, generation=generation)
        robot_id_list_2 = get_top_robot_ids(csv_path_2, top_n=top_n, generation=generation)

        for i in range(top_n):

            robot_id_1 = robot_id_list_1[i]
            robot_id_2 = robot_id_list_2[i]

            robot_save_path_1 = get_robot_save_path(population_path_1, robot_id_1, generation=generation)
            robot_save_path_2 = get_robot_save_path(population_path_2, robot_id_2, generation=generation)

            print(f"Robot 1 generation {generation} top {i+1}: {robot_save_path_1}")
            print(f"Robot 2 generation {generation} top {i+1}: {robot_save_path_2}")

            body_1 = np.load(os.path.join(robot_save_path_1, "body.npy"))
            body_2 = np.load(os.path.join(robot_save_path_2, "body.npy"))

            img_1 = get_image(body_1)
            img_2 = get_image(body_2)

            cv2.imwrite(os.path.join(output_dir_1, f"generation{generation}_top{i+1}_id{robot_id_1}.png"), img_1)
            cv2.imwrite(os.path.join(output_dir_2, f"generation{generation}_top{i+1}_id{robot_id_2}.png"), img_2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default=None, help="Path to the experiment directory")
    parser.add_argument("--generations", type=int, nargs="+", default=None, help="List of generations to visualize")
    parser.add_argument("--top-n", type=int, default=4, help="Number of top robots to visualize")

    args = parser.parse_args()

    get_best_robot_images(args.experiment_dir, generations=args.generations, top_n=args.top_n)
