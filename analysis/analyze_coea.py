import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

if __name__ == "__main__":

    df1 = pd.read_csv("./experiments/coea/Sumo-v0/ibalab-server2/robot_1/fitnesses.csv")
    df2 = pd.read_csv("./experiments/coea/Sumo-v0/ibalab-server2/robot_2/fitnesses.csv")

    df1.drop(columns=["generation"], inplace=True)
    df2.drop(columns=["generation"], inplace=True)

    max_fitnesses_1 = df1.max(axis=1)
    median_fitnesses_1 = df1.median(axis=1)

    max_fitnesses_2 = df2.max(axis=1)
    median_fitnesses_2 = df2.median(axis=1)

    plt.plot(median_fitnesses_1, label="Robot 1 Median", color="dodgerblue")
    plt.plot(max_fitnesses_1, label="Robot 1 Max", linestyle="--", color="dodgerblue")

    plt.plot(median_fitnesses_2, label="Robot 2 Median", color="orange")
    plt.plot(max_fitnesses_2, label="Robot 2 Max", linestyle="--", color="orange")

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Median and Max Fitness for Robot 1 and Robot 2")
    plt.legend()
    plt.savefig("./analysis/figures/coea_fitnesses.png")
