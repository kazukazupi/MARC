from typing import Any, Dict

import numpy as np

from utils import AgentID

ObsType = np.ndarray
ActionType = np.ndarray

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]
RewardDict = Dict[AgentID, float]
BoolDict = Dict[AgentID, bool]
InfoDict = Dict[AgentID, Dict[str, Any]]
