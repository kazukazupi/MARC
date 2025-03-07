import argparse
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch

from alg.coea.structure import Structure
from analysis.analysis_utils import get_max_generation, get_robot_save_path, get_top_robot_ids


def write_csv(experiment_path: str, csv_path: str):

    agent_names = ["robot_1", "robot_2"]
    max_generation = get_max_generation(os.path.join(experiment_path, agent_names[0]))

    # generations = [0, 5, 10]

    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["generation1", "generation2", "return1", "return2"])

    # robot_1 loop
    for generation_1 in range(max_generation + 1):
        # for generation_1 in generations:
        top_robot_ids_1 = get_top_robot_ids(
            os.path.join(experiment_path, agent_names[0], "fitnesses.csv"), top_n=3, generation=generation_1
        )

        # robot_2 loop
        for generation_2 in range(max_generation + 1):
            # for generation_2 in generations:
            top_robot_ids_2 = get_top_robot_ids(
                os.path.join(experiment_path, agent_names[1], "fitnesses.csv"), top_n=3, generation=generation_2
            )

            # robot_1 vs robot_2 loop
            for robot_id_1 in top_robot_ids_1:
                for robot_id_2 in top_robot_ids_2:
                    robot_save_path_1 = get_robot_save_path(
                        os.path.join(experiment_path, agent_names[0]), robot_id_1, generation_1
                    )
                    robot_save_path_2 = get_robot_save_path(
                        os.path.join(experiment_path, agent_names[1]), robot_id_2, generation_2
                    )

                    structure_1 = Structure.from_save_path(robot_save_path_1)
                    structure_2 = Structure.from_save_path(robot_save_path_2)

                    from evaluate import evaluate

                    results = evaluate(
                        {"robot_1": structure_1, "robot_2": structure_2},
                        "Sumo-v0",
                        1,
                        torch.device("cpu"),
                    )
                    fitness_1 = results["robot_1"]
                    fitness_2 = results["robot_2"]

                    with open(csv_path, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([generation_1, generation_2, fitness_1, fitness_2])


def draw_heatmap(csv_path: str):

    plt.rcParams["font.family"] = "Times New Roman"

    # df = pd.read_csv(csv_path)
    # df1 = df.groupby("generation1").mean().reset_index()
    # df2 = df.groupby("generation2").mean().reset_index()

    # fig, ax = plt.subplots(figsize=(8, 6))

    # ax.plot(df1["generation1"], df1["return1"], label="Generation 1 vs Return 1")
    # ax.plot(df2["generation2"], df2["return2"], label="Generation 2 vs Return 2")

    # ax.set_xlabel("Generation")
    # ax.set_ylabel("Return")
    # ax.set_title("Generation vs Return")
    # ax.legend()

    # plt.tight_layout()
    # plt.show()
    df = pd.read_csv(csv_path)
    df["return_diff"] = df["return1"] - df["return2"]
    df = df.groupby(["generation1", "generation2"]).mean().reset_index()

    heatmap_data = df.pivot_table(index="generation1", columns="generation2", values="return_diff")

    sns.heatmap(heatmap_data, annot=False, cmap="coolwarm_r", center=0)
    plt.title
    plt.xlabel("Generations (Population R)")
    plt.ylabel("Generations (Population L)")
    plt.tight_layout()
    plt.savefig("./analysis/figures/heatmap.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["write", "draw"])
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--csv-path", type=str, default="./analysis/logs/heatmap_log.csv")
    args = parser.parse_args()

    if args.mode == "write":
        write_csv(args.save_path, args.csv_path)
    elif args.mode == "draw":
        draw_heatmap(args.csv_path)
    else:
        raise ValueError
