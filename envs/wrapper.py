from copy import copy

import numpy as np
from pettingzoo import ParallelEnv


class MultiAgentEnvWrapper:

    def __init__(self, env: ParallelEnv):

        self.env = env
        self.agents = copy(env.possible_agents)

    def reset(self):
        observations, _ = self.env.reset()
        observations = {a: np.expand_dims(observations[a], axis=0) for a in self.agents}
        return observations

    def step(self, actions):

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        assert len(set(terminations.values())) == 1, "Not all terminations have the same value"
        assert len(set(truncations.values())) == 1, "Not all truncations have the same value"

        dones = {}

        for a in self.agents:
            dones[a] = terminations[a] or truncations[a]
            infos[a]["TimeLimit.truncated"] = truncations[a] and not terminations[a]

        assert len(set(dones.values())) == 1, "Not all dones have the same value"

        if all(dones.values()):
            for a in self.agents:
                infos[a]["terminal_observation"] = observations[a]
            observations, _ = self.env.reset()

        for a in self.agents:
            observations[a] = np.expand_dims(observations[a], axis=0)
            rewards[a] = np.expand_dims(rewards[a], axis=0)
            dones[a] = np.expand_dims(dones[a], axis=0)

        return observations, rewards, dones, infos

    def close(self):
        self.env.close()
