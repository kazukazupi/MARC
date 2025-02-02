import os
from typing import Any, Dict, Optional

import gymnasium as gym  # type: ignore
import torch
from stable_baselines3.common.monitor import Monitor  # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper  # type: ignore
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize  # type: ignore


class VecPyTorch(VecEnvWrapper):
    """
    環境の入出力をPyTorchのテンソルで扱うためのラッパークラス
    """

    def __init__(self, venv: VecNormalize, device: torch.device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def render(self):
        self.venv.envs[0].render()


def make_env(
    env_name: str,
    rank: int,
    log_dir: str,
    **kwargs: Optional[Dict[str, Any]],
):

    def _thunk():
        env = gym.make(env_name, **kwargs)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=False)
        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    num_processes: int,
    gamma: Optional[float],
    device: torch.device,
    training: bool = True,
    seed: Optional[int] = None,
    **env_kwargs: Optional[Dict[str, Any]],
):
    """
    観測および報酬の正規化、環境への入出力のPyTorchテンソルへの変換の機能を備えた並列環境を作成する関数

    Parameters
    ----------
    env_name : str
        作成する環境の名前
    num_processes : int
        並列に実行する環境の数
    gamma : Optional[float]
        割引率
    device : torch.device
        PyTorchのデバイス（CPUまたはGPU）
    training : bool, optional
        `VecNormalize`において`obs_rms`の更新と報酬の正規化を行うかどうかを指定するフラグ (デフォルトはTrue)
    seed : Optional[int], optional
        環境のシード値

    Returns
    -------
    VecPyTorch
        観測および報酬の正規化が行われ、入出力がPyTorchのテンソルに変換された並列環境
    """

    # envs = [lambda: gym.make(env_name, render_mode=render_mode) for _ in range(num_processes)]
    envs = [make_env(env_name, rank, "logs", **env_kwargs) for rank in range(num_processes)]

    # envsをVectorized Environment（複数の環境を並列に実行する環墋）に変換
    vec_env: VecEnv
    if len(envs) > 1:
        vec_env = SubprocVecEnv(envs)
    else:
        vec_env = DummyVecEnv(envs)

    # 観測および報酬の正規化
    if training:
        assert gamma is not None, "gamma must be provided during training"
        vec_env = VecNormalize(vec_env, gamma=gamma)
    else:
        if env_kwargs.get("render_mode", "") == "human":
            assert num_processes == 1, "render_mode='human' is only supported for a single environment"
        vec_env = VecNormalize(vec_env, norm_reward=False)

    vec_env.seed(seed)
    vec_env.obs_rms

    # 環境の入出力をPyTorchのテンソルに変換
    vec_env = VecPyTorch(vec_env, device)

    return vec_env
