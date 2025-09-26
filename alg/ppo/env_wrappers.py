from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym  # type: ignore
import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd  # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore

from alg.ppo.model import Agent
from envs import make
from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, AgentID, ObsDict, ObsType

VecObsDict = ObsDict
VecActionDict = ActionDict
VecRewardDict = Dict[AgentID, np.ndarray]
VecDoneDict = Dict[AgentID, np.ndarray]
VecInfoDict = Dict[AgentID, List[Dict[str, Any]]]

VecPtObsDict = Dict[AgentID, torch.Tensor]
VecPtActionDict = Dict[AgentID, torch.Tensor]
VecPtRewardDict = Dict[AgentID, torch.Tensor]
VecPtDoneDict = Dict[AgentID, torch.Tensor]
VecPtInfoDict = VecInfoDict


class FixedOpponentEnv(gym.Env):

    def __init__(
        self, env: MultiAgentEvoGymBase, self_id: AgentID, opponent_id: AgentID, opponent: Optional[Agent] = None
    ):

        self.env = env
        self.self_id = self_id
        self.opponent_id = opponent_id
        self.opponent = opponent

        self.observation_space = self.env.observation_space(self_id)
        self.action_space = self.env.action_space(self_id)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        if self.opponent is None:
            ret = self.env.step({self.self_id: action})
        else:
            raise NotImplementedError("FixedOpponentEnv with opponent model is not implemented yet.")

        return (
            ret[0][self.self_id],
            ret[1][self.self_id],
            ret[2][self.self_id],
            ret[3][self.self_id],
            ret[4][self.self_id],
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        if self.opponent is None:
            ret = self.env.reset(seed=seed, options=options)
        else:
            raise NotImplementedError("FixedOpponentEnv with opponent model is not implemented yet.")

        return ret[0][self.self_id], ret[1][self.self_id]

    def render(self):
        return self.env.render()


class VecPytorch:

    def __init__(self, env: VecNormalize, device: torch.device = torch.device("cpu")):
        self.env = env
        self.device = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:

        action_ = action.cpu().numpy()
        observation_, reward_, done_, info = self.env.step(action_)

        observation = torch.tensor(observation_, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward_, dtype=torch.float32).to(self.device)
        done = torch.tensor(done_, dtype=torch.bool).to(self.device)

        return observation, reward, done, info

    def reset(self) -> torch.Tensor:
        observation = self.env.reset()
        return torch.tensor(observation, dtype=torch.float32).to(self.device)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self.env.action_space

    @property
    def obs_rms(self) -> RunningMeanStd:
        return self.env.obs_rms

    @obs_rms.setter
    def obs_rms(self, value: RunningMeanStd) -> None:
        self.env.obs_rms = value

    def get_original_reward(self) -> np.ndarray:
        return self.env.get_original_reward()


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

    def observation_space(self, agent):
        return self.envs[0].observation_space(agent)

    def action_space(self, agent):
        return self.envs[0].action_space(agent)

    def render(self):
        assert self.num_envs == 1, "Rendering is only supported for one environment."
        return self.envs[0].render()


# TODO: テストを行う必要あり（visualize_envとreturnが一致しない）
class MultiAgentVecNormalize(MultiAgentDummyVecEnv):

    def __init__(
        self,
        env_funcs: List[Callable[[], MultiAgentEvoGymBase]],
        training: Union[bool, Dict[AgentID, bool]] = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):

        super().__init__(env_funcs)

        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        if self.norm_obs:
            self.obs_rms_dict = {a: RunningMeanStd(shape=self.envs[0].observation_space(a).shape) for a in self.agents}

        if self.norm_reward:
            self.ret_rms_dict = {a: RunningMeanStd(shape=()) for a in self.agents}

        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.returns = {a: np.zeros(self.num_envs) for a in self.agents}
        self.gamma = gamma
        self.epsilon = epsilon

        if isinstance(training, bool):
            self.training_dict = {a: training for a in self.agents}
        else:
            self.training_dict = training

        self.episode_rewards = {a: np.zeros(self.num_envs) for a in self.agents}

    def step(self, actions: ActionDict) -> Tuple[ObsDict, VecRewardDict, VecDoneDict, VecInfoDict]:

        observations, rewards, dones, infos = super().step(actions)

        for a in self.agents:

            self.episode_rewards[a] += rewards[a]

            if self.norm_obs:
                if self.training_dict[a]:
                    self.obs_rms_dict[a].update(observations[a])
                observations[a] = self._normalize_obs(observations[a], self.obs_rms_dict[a])

                for env_idx in range(self.num_envs):
                    if "terminal_observation" in infos[a][env_idx]:
                        infos[a][env_idx]["terminal_observation"] = self._normalize_obs(
                            infos[a][env_idx]["terminal_observation"], self.obs_rms_dict[a]
                        )

            if self.norm_reward:
                if self.training_dict[a]:
                    self.returns[a] = self.returns[a] * self.gamma + rewards[a]
                    self.ret_rms_dict[a].update(self.returns[a])
                rewards[a] = self._normalize_reward(rewards[a], self.ret_rms_dict[a])

            self.returns[a][dones[a]] = 0.0

            for env_idx, done in enumerate(dones[a]):
                if done:
                    infos[a][env_idx]["episode"] = {"r": self.episode_rewards[a][env_idx]}
                    self.episode_rewards[a][env_idx] = 0.0

        return observations, rewards, dones, infos

    def reset(self) -> ObsDict:

        observations = super().reset()

        for a in self.agents:
            if self.norm_obs:
                if self.training_dict[a]:
                    self.obs_rms_dict[a].update(observations[a])
                observations[a] = self._normalize_obs(observations[a], self.obs_rms_dict[a])

        return observations

    def _normalize_obs(self, obs: ObsType, obs_rms: RunningMeanStd) -> ObsType:
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _normalize_reward(self, reward: np.ndarray, ret_rms: RunningMeanStd) -> np.ndarray:
        return np.clip(reward / np.sqrt(ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)


class MultiAgentVecPytorch:
    def __init__(self, env: MultiAgentVecNormalize, device: torch.device = torch.device("cpu")):
        self.env = env
        self.device = device

    def step(self, actions: VecPtActionDict) -> Tuple[VecPtObsDict, VecPtRewardDict, VecPtDoneDict, VecPtInfoDict]:
        actions_ = {a: action.cpu().numpy() for a, action in actions.items()}
        observations_, rewards_, dones_, infos = self.env.step(actions_)

        observations = {a: torch.tensor(obs, dtype=torch.float32).to(self.device) for a, obs in observations_.items()}
        rewards = {a: torch.tensor(rew, dtype=torch.float32).to(self.device) for a, rew in rewards_.items()}
        dones = {a: torch.tensor(done, dtype=torch.bool).to(self.device) for a, done in dones_.items()}

        return observations, rewards, dones, infos

    def reset(self) -> VecPtObsDict:
        observations = self.env.reset()
        return {a: torch.tensor(obs, dtype=torch.float32).to(self.device) for a, obs in observations.items()}

    def __getattr__(self, name):
        return getattr(self.env, name)


# TODO: 完全な並列環境、シード値設定
def make_multi_agent_vec_envs(
    env_name: str,
    num_processes: int,
    gamma: Optional[float],
    device: torch.device,
    training: Union[bool, Dict[AgentID, bool]] = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    seed: Optional[int] = None,
    **env_kwargs: Any,
):

    def _thunk():
        env = make(env_name, **env_kwargs)
        return env

    if num_processes != 1:
        raise NotImplementedError("Only one process is supported for now.")

    envs = [_thunk for _ in range(num_processes)]

    if (isinstance(training, bool) and training) or (isinstance(training, dict) and any(training.values())):
        assert gamma is not None, "gamma must be provided for training"
        vec_env = MultiAgentVecNormalize(envs, training, norm_obs, norm_reward, gamma=gamma)
    else:
        vec_env = MultiAgentVecNormalize(envs, training, norm_obs, norm_reward)

    for a in vec_env.agents:
        vec_env.action_space(a).seed(seed)

    return MultiAgentVecPytorch(vec_env, device=device)


def make_single_agent_vec_env(
    env_name: str,
    num_processes: int,
    gamma: Optional[float],
    device: torch.device,
    self_id: AgentID,
    opponent_id: AgentID,
    opponent: Optional[Agent] = None,
    training: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    seed: Optional[int] = None,
    **env_kwargs: Any,
) -> VecPytorch:

    def _thunk():
        env = make(env_name, **env_kwargs)
        env = FixedOpponentEnv(env, self_id, opponent_id, opponent)
        return env

    if num_processes != 1:
        raise NotImplementedError("Only one process is supported for now.")

    envs = [_thunk for _ in range(num_processes)]

    venv = DummyVecEnv(envs)

    vec_norm = VecNormalize(
        venv,
        training=training,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        gamma=gamma if gamma is not None else 0.99,
    )

    return VecPytorch(vec_norm, device=device)
