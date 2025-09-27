from typing import List, Literal

AgentID = Literal["robot_1", "robot_2"]

AGENT_1: AgentID = "robot_1"
AGENT_2: AgentID = "robot_2"
AGENT_IDS: List[AgentID] = ["robot_1", "robot_2"]


def get_opponent_id(agent_id: AgentID) -> AgentID:
    if agent_id == AGENT_1:
        return AGENT_2
    elif agent_id == AGENT_2:
        return AGENT_1
    else:
        raise ValueError(f"Invalid agent_id: {agent_id}")
