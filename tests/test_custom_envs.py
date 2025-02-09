from evogym import sample_robot
from pettingzoo.test import parallel_api_test

from envs.sumo_env import SimpleSumoEnvClass


def test_sumo_v0():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = SimpleSumoEnvClass(
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)
