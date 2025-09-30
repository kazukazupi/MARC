import json
import os
from typing import Dict, Union

import numpy as np

from alg.coea.structure import DummyRobotStructure, Structure
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

    structures: Dict[AgentID, Union[Structure, DummyRobotStructure]] = {}
    for a in AGENT_IDS:
        body = np.load(os.path.join("./hand_designed_robots", args.env_name, a, "body.npy"))
        connections = np.load(os.path.join("./hand_designed_robots", args.env_name, a, "connections.npy"))
        structures[a] = Structure(os.path.join(save_path, a), body, connections)

    train(args, structures)
