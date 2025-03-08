import shutil

import numpy as np

from alg.coea.population import Population


def test_population_load():

    try:
        agent_name = "test"
        save_path = "tests/temporary"
        pop_size = 10
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

            population.fitnesses = np.random.rand(pop_size)
            population.update(2, 3)

    finally:
        shutil.rmtree(save_path)
