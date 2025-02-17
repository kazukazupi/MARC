import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    df1 = pd.read_csv(os.path.join(args.save_path, "robot_1", "train_log.csv"))
    df2 = pd.read_csv(os.path.join(args.save_path, "robot_2", "train_log.csv"))

    plt.plot(df1["num timesteps"], df1["train reward"], label="robot_1")
    plt.plot(df2["num timesteps"], df2["train reward"], label="robot_2")
    plt.xlabel("num timesteps")
    plt.ylabel("train reward")
    plt.legend()

    plt.savefig("./analysis/figures/train.png")
    plt.close()
