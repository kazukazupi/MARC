import pickle

import gym  # type:ignore
import neat  # type:ignore

from neat_controller_train import structure_1, structure_2
from utils import test

if __name__ == "__main__":

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "./neat_config/controller_config",
    )

    with open("winner.pkl", "rb") as f:
        winner_net = pickle.load(f)

    pop = neat.Checkpointer.restore_checkpoint("neat-checkpoint-141")
    winner_net = neat.nn.FeedForwardNetwork.create(pop.population.get(16271), config)

    env = gym.make("Sumo-v0", structure_1=structure_1, structure_2=structure_2)
    cum_reward = test(env, winner_net, render=True)
    print(cum_reward)
