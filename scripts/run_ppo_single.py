import json
import os

import numpy as np

from alg.coea.structure import DummyRobotStructure, Structure
from alg.ppo import train
from utils import get_args, get_opponent_id

if __name__ == "__main__":

    args = get_args()

    save_path = os.path.join("experiments", "ppo_single", args.env_name, args.exp_dirname)
    os.makedirs(save_path)

    opponent_robot_id = args.dummy_target
    self_robot_id = get_opponent_id(opponent_robot_id)

    dummy_body_type = args.dummy_body_type

    with open(os.path.join(save_path, "env_info.json"), "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "self_robot_id": self_robot_id,
                "dummy_body_type": dummy_body_type,
            },
            f,
            indent=2,
        )

    body = np.load(os.path.join("./hand_designed_robots", args.env_name, self_robot_id, "body.npy"))
    connections = np.load(os.path.join("./hand_designed_robots", args.env_name, self_robot_id, "connections.npy"))
    self_structure = Structure(os.path.join(save_path, self_robot_id), body, connections)
    opponent_structure = DummyRobotStructure(dummy_body_type)

    train(args, {self_robot_id: self_structure, opponent_robot_id: opponent_structure})
