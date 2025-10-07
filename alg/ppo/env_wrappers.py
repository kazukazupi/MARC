from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym  # type: ignore
import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd  # type: ignore
from stable_baselines3.common.vec_env import VecNormalize  # type: ignore

from envs import make
from envs.base import MultiAgentEvoGymBase
from envs.typehints import ActionDict, ObsDict, ObsType
from utils import AgentID

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

        actions_ = [{key: val[env_idx] for key, val in actions.items()} for env_idx in range(self.num_envs)]

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
    use_pytorch_wrapper: bool = True,
    **env_kwargs: Any,
):
    """
    マルチエージェント用のベクトル化環境を作成する

    Parameters
    ----------
    env_name : str
        環境名
    num_processes : int
        並列環境数
    gamma : Optional[float]
        割引率（training時に必要）
    device : torch.device
        使用するデバイス（use_pytorch_wrapper=Trueの時のみ使用）
    training : Union[bool, Dict[AgentID, bool]]
        訓練モードかどうか
    norm_obs : bool
        観測値を正規化するか
    norm_reward : bool
        報酬を正規化するか
    seed : Optional[int]
        シード値（未実装）
    use_pytorch_wrapper : bool
        PyTorchラッパーを使用するか（学習時はTrue、評価時はFalse推奨）
    **env_kwargs : Any
        環境固有のパラメータ

    Returns
    -------
    Union[MultiAgentVecPytorch, MultiAgentVecNormalize]
        use_pytorch_wrapper=True: MultiAgentVecPytorch (torch.Tensorベース)
        use_pytorch_wrapper=False: MultiAgentVecNormalize (numpy.ndarrayベース)
    """

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

    # for a in vec_env.agents:
    #     vec_env.action_space(a).seed(seed)

    if use_pytorch_wrapper:
        return MultiAgentVecPytorch(vec_env, device=device)
    else:
        return vec_env
