from envs import make
import numpy as np
from evogym import get_full_connectivity
import neat
import os

from alg.neat.evaluate import evaluate_net


if __name__ == "__main__":
    import sys

    # Get checkpoint path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python play.py <checkpoint_file>")
        print("Example: python play.py neat-checkpoint-10")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # Load configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'neat.cfg')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Restore from checkpoint
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    # Get the best genome from the population
    best_genome = None
    best_fitness = float('-inf')
    for genome_id, genome in p.population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

    if best_genome is None:
        print("No genome with fitness found in checkpoint")
        sys.exit(1)

    print(f"Best genome fitness: {best_fitness}")

    # Create network and visualize
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    reward = evaluate_net(net, render_mode="human")
    print(f"Reward: {reward}")
