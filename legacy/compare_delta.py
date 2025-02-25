import numpy as np
import pandas as pd
import seaborn as sns


def get_mean_return_sub(df, delta1, delta2):

    df = df[df["path1"].str.contains(delta1)]
    df = df[df["path2"].str.contains(delta2)]

    mean_return1 = df["return1"].mean()
    mean_return2 = df["return2"].mean()

    return mean_return1, mean_return2


def get_mean_return(df, delta1, delta2):
    sum_return1 = 0
    sum_return2 = 0

    mean_return1, mean_return2 = get_mean_return_sub(df, delta1, delta2)
    sum_return1 += mean_return1
    sum_return2 += mean_return2

    mean_return2, mean_return1 = get_mean_return_sub(df, delta2, delta1)
    sum_return1 += mean_return1
    sum_return2 += mean_return2

    return sum_return1 / 2, sum_return2 / 2


df = pd.read_csv("./results.csv")

import matplotlib.pyplot as plt

deltas = ["000", "050", "080", "100"]

results = np.zeros((4, 4))

for i, delta1 in enumerate(deltas):
    for j, delta2 in enumerate(deltas):
        if i == j:
            results[i][j] = 0
            continue
        mean_return1, mean_return2 = get_mean_return(df, delta1, delta2)
        results[i][j] = mean_return1
        results[j][i] = mean_return2

sns.heatmap(results, annot=True, fmt=".2f", cmap="Blues", xticklabels=deltas, yticklabels=deltas)
plt.title("Mean Returns Heatmap")
plt.xlabel("Delta 1")
plt.ylabel("Delta 2")
plt.show()
