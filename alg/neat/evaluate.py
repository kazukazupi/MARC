from envs import make
import numpy as np
from evogym import get_full_connectivity
import neat
import os


def evaluate_net(net, render_mode=None):
    # Load hand-designed robot for robot_1
    body_1 = np.load(os.path.join("./hand_designed_robots", "Sumo-v0", "robot_1", "body.npy"))
    connections_1 = np.load(os.path.join("./hand_designed_robots", "Sumo-v0", "robot_1", "connections.npy"))

    # Create 5x5 box with all elements = 2 for robot_2
    body_2 = np.full((5, 5), 2)
    connections_2 = get_full_connectivity(body_2)

    # Create Sumo environment
    env = make(
        "Sumo-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
        render_mode=render_mode,
    )

    # Run simulation
    obs_dict, _ = env.reset()
    total_reward = 0.0

    while True:
        # Get robot_1 observations and pass to network
        action_1 = net.activate(obs_dict["robot_1"])

        action = {"robot_1": action_1}
        obs_dict, reward, terminated, truncated, info = env.step(action)
        total_reward += reward["robot_1"]

        if render_mode is not None:
            env.render()

        if all(terminated.values()) or all(truncated.values()):
            break

    env.close()
    return total_reward


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return evaluate_net(net)


def run(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Use parallel evaluator (adjust num_workers based on your CPU cores)
    pe = neat.ParallelEvaluator(8, eval_genome)

    # Run for up to 300 generations.
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nEvaluating best genome:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    final_reward = evaluate_net(winner_net)
    print("Final reward: {!r}".format(final_reward))

    return winner


if __name__ == "__main__":
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'neat.cfg')
    print(config_path)
    run(config_path)
