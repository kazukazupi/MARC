import numpy as np
from evogym import sample_robot

from envs import MultiAgentDummyVecEnv, SimpleSumoEnvClass


def test_maenv_wrapper():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    raw_env = SimpleSumoEnvClass(
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    env = SimpleSumoEnvClass(
        body_1=body_1,
        body_2=body_2,
        connections_1=connections_1,
        connections_2=connections_2,
    )

    wrapped_env = MultiAgentDummyVecEnv(env)

    observations, _ = raw_env.reset()
    wrapped_observations = wrapped_env.reset()

    for obs, wrapped_obs in zip(observations.values(), wrapped_observations.values()):
        assert np.array_equal(obs, wrapped_obs[0])

    for _ in range(10000):

        action_1 = raw_env.action_space("robot_1").sample()
        action_2 = raw_env.action_space("robot_2").sample()
        action = {"robot_1": action_1, "robot_2": action_2}

        observations, rewards, terminations, truncations, _ = raw_env.step(action)
        wrapped_observations, wrapped_rewards, wrapped_dones, wrapped_infos = wrapped_env.step(action)

        if all(terminations.values()) or all(truncations.values()):
            for obs, wrapped_info in zip(observations.values(), wrapped_infos.values()):
                assert np.array_equal(obs, wrapped_info["terminal_observation"])
            observations, _ = raw_env.reset()

        for obs, wrapped_obs in zip(observations.values(), wrapped_observations.values()):
            assert np.array_equal(obs, wrapped_obs[0])

        for reward, wrapped_reward in zip(rewards.values(), wrapped_rewards.values()):
            assert np.array_equal(reward, wrapped_reward[0])

        for term, trunc, wrapped_done, wrapped_info in zip(
            terminations.values(), truncations.values(), wrapped_dones.values(), wrapped_infos.values()
        ):
            assert term or trunc == wrapped_done[0]
            assert wrapped_info["TimeLimit.truncated"] == (trunc and not term)
