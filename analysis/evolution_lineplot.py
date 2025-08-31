import argparse
import csv
import glob
import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch

from alg.coea.structure import Structure
from analysis.analysis_utils import get_max_generation, get_robot_save_path, get_top_robot_ids
from evaluate import evaluate


def write_csv(env_name: str, csv_path: str):
    """
    Evaluates top robots from different experiments and generations in a given environment,
    and writes their results to a CSV file.

    Args:
        env_name (str): Environment name.
        csv_path (str): Output CSV file path.
    """

    exp_dir_list = glob.glob(os.path.join("experiments", "coea", env_name, "*"))
    max_generation = get_max_generation(os.path.join(exp_dir_list[0], "robot_1"))
    agent_names = ["robot_1", "robot_2"]

    # Setup csv
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["exp1", "exp2", "rank1", "rank2", "generation1", "generation2", "return1", "return2"])

    # Loop for all generations, all runs
    for (gen1, gen2), (exp1, exp2) in product(
        product(range(max_generation + 1), repeat=2), product(exp_dir_list, repeat=2)
    ):
        # Get top robot ids
        fitness_csv_path_1 = os.path.join(exp1, agent_names[0], "fitnesses.csv")
        fitness_csv_path_2 = os.path.join(exp2, agent_names[1], "fitnesses.csv")

        top_robot_ids_1 = get_top_robot_ids(fitness_csv_path_1, top_n=3, generation=gen1)
        top_robot_ids_2 = get_top_robot_ids(fitness_csv_path_2, top_n=3, generation=gen2)

        # Loop through top 1 robots vs top n robots
        for (r_1, robot_id_1), (r_2, robot_id_2) in product(enumerate(top_robot_ids_1), enumerate(top_robot_ids_2)):

            if r_1 != 0 and r_2 != 0:
                continue

            # Load Structures
            robot_save_path_1 = get_robot_save_path(os.path.join(exp1, agent_names[0]), robot_id_1, gen1)
            robot_save_path_2 = get_robot_save_path(os.path.join(exp2, agent_names[1]), robot_id_2, gen2)

            structure_1 = Structure.from_save_path(robot_save_path_1)
            structure_2 = Structure.from_save_path(robot_save_path_2)

            # Evaluate
            results = evaluate(
                {"robot_1": structure_1, "robot_2": structure_2},
                env_name,
                1,
                torch.device("cpu"),
            )
            fitness_1 = results["robot_1"]
            fitness_2 = results["robot_2"]

            # Write results to CSV
            with open(csv_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        os.path.basename(exp1),
                        os.path.basename(exp2),
                        r_1,
                        r_2,
                        gen1,
                        gen2,
                        fitness_1,
                        fitness_2,
                    ]
                )


def draw_lineplot(csv_path: str, pdf_path: str):

    df = pd.read_csv(csv_path)
    df1 = (
        df[df["rank2"] == 0][["generation1", "exp1", "return1"]].groupby(["generation1", "exp1"]).mean().reset_index()
    )
    df2 = (
        df[df["rank1"] == 0][["generation2", "exp2", "return2"]].groupby(["generation2", "exp2"]).mean().reset_index()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df1,
        x="generation1",
        y="return1",
        marker="o",
        label="Left",
        errorbar=("ci", 95),
    )

    sns.lineplot(
        data=df2,
        x="generation2",
        y="return2",
        marker="o",
        label="Right",
        errorbar=("ci", 95),
    )

    plt.xlabel("Generation")
    plt.ylabel("Return")
    plt.title("Evolution Lineplot")
    plt.legend(title="Population")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


def draw_all_lineplot(pdf_path: str):

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 21

    envs = ["Sumo-v0", "MultiPusher-v0", "MultiPusher-v2", "Ojama-d4", "ChimneyClash"]
    show_env_dict = {
        "Sumo-v0": "Sumo",
        "MultiPusher-v0": "BoxPush",
        "MultiPusher-v2": "AboveBoxPush",
        "Ojama-d4": "PassAndBlock",
        "ChimneyClash": "ChimneyClash",
    }
    data = {}

    for env in envs:
        csv_path = os.path.join("analysis", "logs", "lineplot", f"{env}.csv")
        df = pd.read_csv(csv_path)
        df1 = (
            df[df["rank1"] == 0][["generation1", "exp1", "return1"]]
            .groupby(["generation1", "exp1"])
            .mean()
            .reset_index()
        )
        df2 = (
            df[df["rank2"] == 0][["generation2", "exp2", "return2"]]
            .groupby(["generation2", "exp2"])
            .mean()
            .reset_index()
        )
        data[env] = {"df1": df1, "df2": df2}

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    for i, (ax, (env, env_data)) in enumerate(zip(axes, data.items())):
        sns.lineplot(
            data=env_data["df1"],
            x="generation1",
            y="return1",
            marker="o",
            label="Left",
            errorbar=("ci", 95),
            ax=ax,
        )
        sns.lineplot(
            data=env_data["df2"],
            x="generation2",
            y="return2",
            marker="o",
            label="Right",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(show_env_dict[env])
        ax.set_xlabel("Generations")
        if i == 0:
            ax.set_ylabel("Robot Performance")
        else:
            ax.set_ylabel("")
        # ax.legend().remove()
        if i == 0:
            pass
            # ax.legend(title="Population", loc="lower center", ncol=2,)
            # ax.legend()
        else:
            ax.legend().remove()
    # fig.legend(title="Population", loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # plt.subplots_adjust(wspace=0.3, hspace=1.0)

    # fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    # plt.subplots_adjust(top=1.30, bottom=1.20)

    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["write", "draw", "draw_all"],
        required=True,
        help="Operation mode: write CSV, draw lineplot for a specific env, or draw all envs",
    )
    parser.add_argument("--env-name", type=str, help="Name of the environment")
    args = parser.parse_args()

    if args.mode in ["write", "draw"]:
        if args.env_name is None:
            raise ValueError("env_name must be specified in write or draw mode")
        csv_path = os.path.join("analysis", "logs", "lineplot", f"{args.env_name}.csv")
        pdf_path = os.path.join("analysis", "figures", "lineplot", f"{args.env_name}.pdf")
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
    elif args.mode == "draw_all":
        pdf_path = os.path.join("analysis", "figures", "lineplot", "all.pdf")

    if not os.path.exists(os.path.dirname(pdf_path)):
        os.makedirs(os.path.dirname(pdf_path))

    if args.mode == "write":
        write_csv(args.env_name, csv_path)
    elif args.mode == "draw":
        draw_lineplot(csv_path, pdf_path)
    elif args.mode == "draw_all":
        draw_all_lineplot(pdf_path)
    else:
        raise ValueError
