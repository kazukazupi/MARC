import random
from typing import Dict, List


def get_matches(pop_size: int, num_opponents: int, agent_names: List[str]) -> List[Dict[str, int]]:
    indices = list(range(pop_size))
    random.shuffle(indices)
    matching = []
    for a in range(pop_size):
        for d in range(num_opponents):
            b = indices[(a + d) % pop_size]
            matching.append(
                {
                    agent_names[0]: a,
                    agent_names[1]: b,
                }
            )
    return matching


def get_percent_survival_evals(curr_train: int, max_trains: int) -> float:
    low = 0.0
    high = 0.6
    return ((max_trains - curr_train - 1) / (max_trains - 1)) * (high - low) + low


if __name__ == "__main__":
    pop_size = 3
    num_opponents = 2
    matching = get_matches(pop_size, num_opponents, ["agent1", "agent2"])
    print(matching)
    print(len(matching))
