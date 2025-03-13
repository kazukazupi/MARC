import random
import shutil

import numpy as np

from alg.coea.population import Population


def test_scores():

    agent_name = "test"
    save_path = "tests/tmp_scores"
    pop_size = 25
    eval_num_opponents = 15
    robot_shape = (5, 5)

    try:
        population = Population(agent_name, save_path, pop_size, robot_shape)

        scores = np.random.rand(pop_size, eval_num_opponents)

        for i in range(pop_size):
            for j in range(eval_num_opponents):
                population.set_score(i, j, scores[i, j])

        for i in range(pop_size):
            assert population.fitnesses[i] == np.mean(scores[i])

        to_delete_indices = [10, 13]
        for i in to_delete_indices:
            population.delete_score(i)
        scores = np.delete(scores, to_delete_indices, axis=1)

        for i in range(pop_size):
            assert population.fitnesses[i] == np.mean(scores[i])

    finally:
        shutil.rmtree(save_path)
        pass


def test_population_load():

    try:
        agent_name = "test"
        save_path = "tests/temporary"
        pop_size = 10
        num_survivors = 2
        robot_shape = (5, 5)

        population = Population(agent_name, save_path, pop_size, robot_shape)

        for _ in range(10):

            loaded_population = Population(agent_name, save_path, pop_size, robot_shape, True)

            assert population.agent_name == loaded_population.agent_name
            assert population.save_path == loaded_population.save_path
            assert population.csv_path == loaded_population.csv_path
            assert population.generation == loaded_population.generation
            assert population.population_structure_hashes == loaded_population.population_structure_hashes
            for structure, loaded_structure in zip(population.structures, loaded_population.structures):
                assert structure.save_path == loaded_structure.save_path
                assert np.array_equal(structure.body, loaded_structure.body)
                assert np.array_equal(structure.connections, loaded_structure.connections)
                assert structure.is_trained == loaded_structure.is_trained
                assert structure.is_died == loaded_structure.is_died
                assert structure.fitness == loaded_structure.fitness

            # set fitnesses
            for i in range(pop_size):
                for j in range(pop_size):
                    population.set_score(i, j, random.random())

            population.update(num_survivors, pop_size - num_survivors)

    finally:
        shutil.rmtree(save_path)
