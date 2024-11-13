import multiprocessing
import pickle

import evogym  # type:ignore
import gym  # type:ignore
import neat  # type:ignore
import numpy as np
from tqdm import tqdm  # type:ignore

import envs
from utils import test

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


def eval_genome(genome, config, _genome_id, _generation):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("Sumo-v0", structure_1=structure_1, structure_2=structure_2)
    cum_reward = test(env, net)
    return cum_reward["robot_1"]


def eval_genomes(genomes, config):
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
    pop.add_reporter(neat.Checkpointer(generation_interval=2))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate_fitness, n=250)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner_net, f)
