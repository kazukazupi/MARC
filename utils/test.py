import gym  # type:ignore
import neat  # type:ignore


def test(env: gym.Env, net: neat.nn.FeedForwardNetwork, render: bool = False):

    obs = env.reset()

    cum_reward = {"robot_1": 0.0, "robot_2": 0.0}

    for i in range(500):

        action_1 = net.activate(obs["robot_1"])
        action_2 = env.action_space.sample()["robot_2"]

        action = {"robot_1": action_1, "robot_2": action_2}

        obs, reward, done, info = env.step(action)

        cum_reward["robot_1"] += reward["robot_1"]
        cum_reward["robot_2"] += reward["robot_2"]
        if render:
            env.render()
        if done:
            env.reset()

    env.close()

    return cum_reward
