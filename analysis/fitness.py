import glob
import os

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 14

if __name__ == "__main__":

    envs = ["Sumo-v0", "MultiPusher-v0", "MultiPusher-v2", "Ojama-d4", "ChimneyClash"]
    env_titles = {
        "Sumo-v0": "Sumo",
        "MultiPusher-v0": "BoxPush",
        "MultiPusher-v2": "AboveBoxPush",
        "Ojama-d4": "PassAndBlock",
        "ChimneyClash": "ChimneyClash",
    }

    fig, axes = plt.subplots(3, 5)
    axes = axes.flatten()

    for j, env in enumerate(envs):

        exp_dir_list = glob.glob(os.path.join("experiments", "coea", env, "*"))
        for i, exp_dir in enumerate(exp_dir_list):
            robot_1_fitness_df = pd.read_csv(os.path.join(exp_dir, "robot_1", "fitnesses.csv"))
            robot_2_fitness_df = pd.read_csv(os.path.join(exp_dir, "robot_2", "fitnesses.csv"))

            robot_1_fitness_df = robot_1_fitness_df.drop(columns=["generation"])
            robot_2_fitness_df = robot_2_fitness_df.drop(columns=["generation"])

            robot_1_fitness = robot_1_fitness_df.max(axis=1)
            robot_2_fitness = robot_2_fitness_df.max(axis=1)

            axes[i * 5 + j].plot(robot_1_fitness, label="Left")
            axes[i * 5 + j].plot(robot_2_fitness, label="Right")
            if i == 0:
                axes[i * 5 + j].set_title(env_titles[env])
            if i == 2:
                axes[i * 5 + j].set_xlabel("Generations")
            if j == 0:
                axes[i * 5 + j].set_ylabel("Fitness")
            if j == 4 and i == 2:
                axes[i * 5 + j].legend()
            # axes[i*5 + j].set_title(f"{env} - Exp {i+1}")
            # axes[j * 3 + i].legend()

    plt.tight_layout()
    plt.show()
