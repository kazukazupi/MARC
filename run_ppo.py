import json
import os
from typing import Dict

import numpy as np

from alg.coea.structure import Structure
from alg.ppo import train
from utils import AGENT_IDS, AgentID, get_args

if __name__ == "__main__":

    args = get_args()

    save_path = os.path.join("experiments", "ppo", args.env_name, args.exp_dirname)
    os.makedirs(save_path)
    with open(os.path.join(save_path, "env_info.json"), "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "agents": AGENT_IDS,
            },
            f,
            indent=2,
        )

    structures: Dict[AgentID, Structure] = {}
    for agent_name in AGENT_IDS:
        body = np.load(os.path.join("./hand_designed_robots", args.env_name, agent_name, "body.npy"))
        connections = np.load(os.path.join("./hand_designed_robots", args.env_name, agent_name, "connections.npy"))
        structures[agent_name] = Structure(os.path.join(save_path, agent_name), body, connections)

    train(args, structures)
