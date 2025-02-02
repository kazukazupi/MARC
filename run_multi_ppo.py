import numpy as np
from evogym import sample_robot

from envs import MultiAgentDummyVecEnv, SimpleSumoEnvClass


def main():

    body_1, connections_1 = sample_robot((5, 5))
    body_2, connections_2 = sample_robot((5, 5))

    env_funs = [
        lambda: SimpleSumoEnvClass(
            body_1=body_1,
            body_2=body_2,
            connections_1=connections_1,
            connections_2=connections_2,
        )
    ]

    vec_env = MultiAgentDummyVecEnv(env_funs)

    observations = vec_env.reset()
    print(observations)


if __name__ == "__main__":
    main()
