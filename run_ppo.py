import json
import os

import numpy as np

from alg.coea.structure import Structure
from alg.ppo import train
from utils import get_agent_names, get_args

if __name__ == "__main__":

    args = get_args()
    agent_ids = get_agent_names()

    save_path = os.path.join("experiments", "ppo", args.env_name, args.exp_dirname)
    os.makedirs(save_path)
    with open(os.path.join(save_path, "env_info.json"), "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "agents": agent_ids,
            },
            f,
            indent=2,
        )

    structures = {}
    for agent_id in agent_ids:
        body = np.load(os.path.join("./hand_designed_robots", args.env_name, agent_id, "body.npy"))
        connections = np.load(os.path.join("./hand_designed_robots", args.env_name, agent_id, "connections.npy"))
        structures[agent_id] = Structure(os.path.join(save_path, agent_id), body, connections)

    train(args, structures)
