import gymnasium as gym
from evogym import sample_robot

import envs

if __name__ == "__main__":

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = gym.make(
        "Sumo-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
        render_mode="human",
    )
    env.reset()

    cum_reward_1 = 0.0
    cum_reward_2 = 0.0

    for i in range(2000):
        action = env.action_space.sample()
        ob, reward, terminated, truncated, info = env.step(action)
        cum_reward_1 += reward["robot_1"]
        cum_reward_2 += reward["robot_2"]
        if terminated or truncated:
            env.reset()
            break
    env.close()

    print("Cumulative reward for robot 1: {}".format(cum_reward_1))
    print("Cumulative reward for robot 2: {}".format(cum_reward_2))
