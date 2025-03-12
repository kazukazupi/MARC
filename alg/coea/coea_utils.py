import json
import os
import random
from typing import Dict, List, Literal

from pydantic import BaseModel


def get_matches(
    listA: List[int],
    listB: List[int],
    num_opponents: int,
    agent_names: List[str],
    metadata_dir_path: str,
    generation: int,
    mode: Literal["train", "eval"],
) -> List[Dict[str, int]]:

    assert len(listA) == len(listB), "Lists must be of equal length"
    num_opponents = min(num_opponents, len(listB))

    shuffled_listB = listB.copy()
    random.shuffle(shuffled_listB)
    matching = []

    for idx, a in enumerate(listA):
        for d in range(num_opponents):
            b = shuffled_listB[(idx + d) % len(listB)]
            matching.append(
                {
                    agent_names[0]: a,
                    agent_names[1]: b,
                }
            )

    save_path = os.path.join(metadata_dir_path, f"{mode}_match_metadata.json")
    match_metadata = MatchMetadata(generation=generation, matches=matching)
    with open(save_path, "w") as f:
        json.dump(match_metadata.model_dump(), f, indent=4)

    return matching


def get_percent_survival_evals(curr_train: int, max_trains: int) -> float:
    low = 0.0
    high = 0.6
    return ((max_trains - curr_train - 1) / (max_trains - 1)) * (high - low) + low


class StructureMetadata(BaseModel):
    is_trained: bool
    is_died: bool
    scores: Dict[int, float] = {}


class MatchMetadata(BaseModel):
    generation: int
    matches: List[Dict[str, int]]
