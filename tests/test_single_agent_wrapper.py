import numpy as np
from evogym import get_full_connectivity, sample_robot  # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

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
        # render_mode="human",
    )
    env = FixedOpponentEnv(env, "robot_1")

    env.reset()

    count = 0

    while count < 2:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            count += 1

    env.close()


def test_fixed_opponent_env_vectorize():

    body_1, connections_1 = sample_robot((5, 5))
    body_2 = np.ones((5, 5), dtype=int) * 2
    connections_2 = get_full_connectivity(body_2)

    def _thunk():
        env = make(
            "Sumo-v0",
            body_1=body_1,
            body_2=body_2,
            connections_1=connections_1,
            connections_2=connections_2,
        )
        return FixedOpponentEnv(env, "robot_1")

    vec_env = DummyVecEnv([_thunk])

    vec_env.reset()

    for _ in range(1000):
        action: np.ndarray = vec_env.action_space.sample()
        action = np.expand_dims(action, axis=0)  # type: ignore
        vec_env.step(action)

    vec_env.close()
