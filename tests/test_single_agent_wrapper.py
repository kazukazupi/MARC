import numpy as np
from evogym import get_full_connectivity, sample_robot  # type: ignore

from alg.ppo.env_wrappers import FixedOpponentEnv
from envs import make


def test_fixed_opponent_env():

    body_1, connections_1 = sample_robot((5, 5))
    body_2 = np.ones((5, 5), dtype=int) * 2
    connections_2 = get_full_connectivity(body_2)

    env = make(
        "Sumo-v0",
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
        render_mode="human",
    )
    env = FixedOpponentEnv(env, "robot_1", "robot_2")

    env.reset()

    count = 0

    while count < 2:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            count += 1

    env.close()
