from evogym import sample_robot  # type: ignore
from pettingzoo.test import parallel_api_test  # type: ignore

from envs import make


def test_sumo_v0():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "Sumo-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_multi_pusher_v0():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "MultiPusher-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_multi_pusher_v1():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "WallPusher-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_multi_pusher_v2():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "MultiPusher-v2",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_ojama_d3():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "Ojama-d3",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_ojama_d4():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "Ojama-d4",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_ojama_d5():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "Ojama-d5",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_chimney_clash():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "ChimneyClash",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)
