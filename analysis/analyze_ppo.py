import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    df1 = pd.read_csv("./log4/robot_1/train_log.csv")
    df2 = pd.read_csv("./log4/robot_2/train_log.csv")

    plt.plot(df1["num timesteps"], df1["train reward"], label="robot_1")
    plt.plot(df2["num timesteps"], df2["train reward"], label="robot_2")
    plt.xlabel("num timesteps")
    plt.ylabel("train reward")
    plt.legend()

    plt.savefig("./analysis/figures/train.png")
    plt.close()

    df1 = pd.read_csv("./log4/robot_1/eval_log.csv")
    df2 = pd.read_csv("./log4/robot_2/eval_log.csv")

    plt.plot(df1["num timesteps"], df1["eval rewrard"], label="robot_1")
    plt.plot(df2["num timesteps"], df2["eval rewrard"], label="robot_2")
    plt.xlabel("num timesteps")
    plt.ylabel("eval reward")
    plt.legend()

    plt.savefig("./analysis/figures/eval.png")
    plt.close()
