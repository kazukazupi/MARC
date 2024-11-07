import evogym  # type:ignore
import gym  # type:ignore
import neat  # type:ignore
import numpy as np
from tqdm import tqdm  # type:ignore

import envs

body = np.array(
    [
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
    ]
)

structure_1 = (body.copy(), evogym.get_full_connectivity(body))
structure_2 = (body.copy(), evogym.get_full_connectivity(body))


def test(net: neat.nn.FeedForwardNetwork, render: bool = False):

    env = gym.make("Sumo-v0", structure_1=structure_1, structure_2=structure_2)
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


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    cum_reward = test(net, False)
    return cum_reward["robot_1"]


def eval_genomes(genomes, config, _):
    for genome_id, genome in tqdm(genomes):
        genome.fitness = eval_genome(genome, config)


if __name__ == "__main__":

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "./neat_config/controller_config",
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, n=500)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    cum_reward = test(winner_net, True)
    print(cum_reward)
