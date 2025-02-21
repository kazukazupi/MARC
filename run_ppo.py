import json
import os

import numpy as np

from alg.ppo import train
from utils import get_args

if __name__ == "__main__":

    args = get_args()

    save_path = os.path.join("experiments", "ppo", args.env_name, args.exp_dirname)

    body_1 = np.load(os.path.join("./hand_designed_robots", args.env_name, "robot_1", "body.npy"))
    body_2 = np.load(os.path.join("./hand_designed_robots", args.env_name, "robot_2", "body.npy"))
    connections_1 = np.load(os.path.join("./hand_designed_robots", args.env_name, "robot_1", "connections.npy"))
    connections_2 = np.load(os.path.join("./hand_designed_robots", args.env_name, "robot_2", "connections.npy"))

    os.makedirs(save_path)

    with open(os.path.join(save_path, "env_info.json"), "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "agents": ["robot_1", "robot_2"],
            },
            f,
            indent=2,
        )

    save_path_1 = os.path.join(save_path, "robot_1")
    save_path_2 = os.path.join(save_path, "robot_2")

    train(args, save_path_1, save_path_2, body_1, body_2, connections_1, connections_2)
