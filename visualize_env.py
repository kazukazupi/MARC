import argparse
import random
from typing import Dict

import cv2  # type: ignore
import numpy as np
from evogym import sample_robot  # type: ignore

from envs import make
from utils import AgentID

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-name", type=str, default="Sumo-v0")
    parser.add_argument("--mode", choices=["human", "rgb_array"], default="human")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        args.env_name,
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
        render_mode="rgb_array",
    )

    env.reset()
    frame = env.render()

    cv2.imwrite("./first_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    env.close()

    env = make(
        args.env_name,
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
        render_mode=args.mode,
    )
    env.reset()

    if args.seed is not None:
        env.action_space("robot_1").seed(args.seed)
        env.action_space("robot_2").seed(args.seed)

    cum_reward_1 = 0.0
    cum_reward_2 = 0.0
    timestep = 0

    while True:
        if args.mode == "rgb_array":
            frame = env.render()
            cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        action_1 = env.action_space("robot_1").sample()
        action_2 = env.action_space("robot_2").sample()
        action: Dict[AgentID, np.ndarray] = {"robot_1": action_1, "robot_2": action_2}
        ob, reward, terminated, truncated, info = env.step(action)
        cum_reward_1 += reward["robot_1"]
        cum_reward_2 += reward["robot_2"]
        # if all(truncated.values()):
        if all(terminated.values()) or all(truncated.values()):
            env.reset()
            break
        timestep += 1

    env.close()

    print("Timesteps: {}".format(timestep))
    print("Cumulative reward for robot 1: {}".format(cum_reward_1))
    print("Cumulative reward for robot 2: {}".format(cum_reward_2))
