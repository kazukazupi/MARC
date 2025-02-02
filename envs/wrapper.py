from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from envs import MultiAgentEvoGymBase
from envs.typehints import ActionDict, AgentID, ObsDict, ObsType

VecObsDict = ObsDict
VecActionDict = ActionDict
VecRewardDict = Dict[AgentID, np.ndarray]
VecDoneDict = Dict[AgentID, np.ndarray]
VecInfoDict = Dict[AgentID, List[Dict[str, Any]]]


class MultiAgentDummyVecEnv:

    def __init__(self, env_funs: List[Callable[[], MultiAgentEvoGymBase]]):

        self.envs = [env_fun() for env_fun in env_funs]
        self.num_envs = len(self.envs)

        env = self.envs[0]
        self.agents = copy(env.possible_agents)
        obs_spaces = {a: env.observation_space(a) for a in self.agents}

        self.buf_obs = {
            a: np.zeros((self.num_envs, *obs_spaces[a].shape), dtype=obs_spaces[a].dtype) for a in self.agents
        }
        self.buf_dones = {a: np.zeros((self.num_envs,), dtype=bool) for a in self.agents}
        self.buf_rews = {a: np.zeros((self.num_envs,), dtype=np.float64) for a in self.agents}
        self.buf_infos: VecInfoDict = {a: [{} for _ in range(self.num_envs)] for a in self.agents}

    def reset(self) -> VecObsDict:
        for env_idx in range(self.num_envs):
            observations, _ = self.envs[env_idx].reset()
            for a in self.agents:
                self.buf_obs[a][env_idx] = observations[a]
        return deepcopy(self.buf_obs)

    def step(self, actions: VecActionDict) -> Tuple[VecObsDict, VecRewardDict, VecDoneDict, VecInfoDict]:

        actions_ = [{a: actions[a][env_idx] for a in self.agents} for env_idx in range(self.num_envs)]

        for env_idx in range(self.num_envs):
            observations, rewards, terminations, truncations, infos = self.envs[env_idx].step(actions_[env_idx])
            for a in self.agents:
                self.buf_rews[a][env_idx] = rewards[a]
                self.buf_dones[a][env_idx] = terminations[a] or truncations[a]
                self.buf_infos[a][env_idx] = infos[a]
                self.buf_infos[a][env_idx]["TimeLimit.truncated"] = truncations[a] and not terminations[a]

            if all([self.buf_dones[a][env_idx] for a in self.agents]):
                for a in self.agents:
                    self.buf_infos[a][env_idx]["terminal_observation"] = observations[a]
                observations, _ = self.envs[env_idx].reset()

            for a in self.agents:
                self.buf_obs[a][env_idx] = observations[a]

        return deepcopy(self.buf_obs), deepcopy(self.buf_rews), deepcopy(self.buf_dones), deepcopy(self.buf_infos)

    def close(self):
        for env in self.envs:
            env.close()


class MultiAgentVecNormalize(MultiAgentDummyVecEnv):

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

            for env_idx in range(self.num_envs):
                if "terminal_observation" in infos[a][env_idx]:
                    infos[a][env_idx]["terminal_observation"] = self._normalize_obs(
                        infos[a][env_idx]["terminal_observation"], self.obs_rms_dict[a]
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
