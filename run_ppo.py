import json
import os

import numpy as np

from alg.coea.structure import Structure
from alg.ppo import train
from utils import get_agent_names, get_args

if __name__ == "__main__":

    args = get_args()
    agent_names = get_agent_names()

    save_path = os.path.join("experiments", "ppo", args.env_name, args.exp_dirname)
    os.makedirs(save_path)
    with open(os.path.join(save_path, "env_info.json"), "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "agents": agent_names,
            },
            f,
            indent=2,
        )

    structures = {}
    for agent_name in agent_names:
        body = np.load(os.path.join("./hand_designed_robots", args.env_name, agent_name, "body.npy"))
        connections = np.load(os.path.join("./hand_designed_robots", args.env_name, agent_name, "connections.npy"))
        structures[agent_name] = Structure(os.path.join(save_path, agent_name), body, connections)

    train(args, structures)
