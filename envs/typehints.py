from typing import Any, Dict

import numpy as np

ObsType = np.ndarray
ActionType = np.ndarray

ObsDict = Dict[str, ObsType]
ActionDict = Dict[str, ActionType]
RewardDict = Dict[str, float]
BoolDict = Dict[str, bool]
InfoDict = Dict[str, Dict[str, Any]]
