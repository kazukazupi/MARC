from copy import copy

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from envs import MultiAgentEvoGymBase
from envs.typehints import ActionDict, ObsDict, ObsType


class MultiAgentEnvWrapper:

    def __init__(self, env: MultiAgentEvoGymBase):

        self.env = env
        self.agents = copy(env.possible_agents)

    def reset(self) -> ObsDict:
        observations, _ = self.env.reset()
        observations = {a: np.expand_dims(observations[a], axis=0) for a in self.agents}
        return observations

    def step(self, actions: ActionDict):

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

        observations = {a: np.expand_dims(observations[a], axis=0) for a in self.agents}
        rewards_ = {a: np.expand_dims(rewards[a], axis=0) for a in self.agents}
        dones_ = {a: np.expand_dims(dones[a], axis=0) for a in self.agents}

        return observations, rewards_, dones_, infos

    def close(self):
        self.env.close()


class MultiAgentNormalize(MultiAgentEnvWrapper):

    def __init__(
        self,
        env: MultiAgentEvoGymBase,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):

        super().__init__(env)

        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        if self.norm_obs:
            self.obs_rms_dict = {a: RunningMeanStd(shape=env.observation_space(a).shape) for a in self.agents}

        self.ret_rms_dict = {a: RunningMeanStd(shape=()) for a in self.agents}

        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.returns = {a: np.zeros(1) for a in self.agents}
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

    def step(self, actions: ActionDict):

        observations, rewards, dones, infos = super().step(actions)

        for a in self.agents:

            if self.norm_obs:
                if self.training:
                    self.obs_rms_dict[a].update(observations[a])
                observations[a] = self._normalize_obs(observations[a], self.obs_rms_dict[a])

            if self.norm_reward:
                if self.training:
                    self.returns[a] = self.returns[a] * self.gamma + rewards[a]
                    self.ret_rms_dict[a].update(self.returns[a])
                rewards[a] = self._normalize_reward(rewards[a], self.ret_rms_dict[a])

            if "terminal_observation" in infos[a]:
                infos[a]["terminal_observation"] = self._normalize_obs(
                    infos[a]["terminal_observation"], self.obs_rms_dict[a]
                )

            self.returns[a][dones[a]] = 0.0

    def reset(self) -> ObsDict:

        observations = super().reset()

        for a in self.agents:
            if self.norm_obs:
                if self.training:
                    self.obs_rms_dict[a].update(observations[a])
                observations[a] = self._normalize_obs(observations[a], self.obs_rms_dict[a])

        return observations

    def _normalize_obs(self, obs: ObsType, obs_rms: RunningMeanStd) -> ObsType:
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _normalize_reward(self, reward: np.ndarray, ret_rms: RunningMeanStd) -> np.ndarray:
        return np.clip(reward / np.sqrt(ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
