import gym  # type:ignore
from evogym import sample_robot  # type:ignore

import envs

if __name__ == "__main__":

    structure_1 = sample_robot((5, 5))
    structure_2 = sample_robot((5, 5))

    env = gym.make("Sumo-v0", structure_1=structure_1, structure_2=structure_2)
    env.reset()

    cum_reward_1 = 0.0
    cum_reward_2 = 0.0

    for i in range(500):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        cum_reward_1 += reward["robot_1"]
        cum_reward_2 += reward["robot_2"]
        env.render()
        if done:
            env.reset()
    env.close()

    print("Cumulative reward for robot 1: {}".format(cum_reward_1))
    print("Cumulative reward for robot 2: {}".format(cum_reward_2))
