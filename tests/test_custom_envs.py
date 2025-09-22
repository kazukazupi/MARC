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


def test_box_push_v0():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "BoxPush-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_above_box_push_v0():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "AboveBoxPush-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)


def test_pass_and_block():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env = make(
        "PassAndBlock-v0",
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
        "ChimneyClash-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    parallel_api_test(env, num_cycles=10000)
