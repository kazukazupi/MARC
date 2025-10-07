from typing import Tuple

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd  # type: ignore

from alg.controller import Controller
from alg.ppo.model import Agent


class AgentController(Controller):
    """PPOエージェントのController実装（評価専用）

    このクラスはPPOで訓練されたAgentをController抽象クラスとして
    ラップし、numpy配列ベースの評価インターフェースを提供します。

    Attributes
    ----------
    agent : Agent
        PPOエージェント（学習済み）
    device : torch.device
        計算に使用するデバイス
    """

    def __init__(self, agent: Agent, device: torch.device):
        """
        AgentControllerを初期化する

        Parameters
        ----------
        agent : Agent
            PPOエージェント
        device : torch.device
            計算に使用するデバイス
        """
        self.agent = agent
        self.device = device

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        行動を選択する（評価時用のnumpy版、Controllerインターフェース実装）

        Parameters
        ----------
        observation : np.ndarray
            観測値（shape: (obs_dim,) or (batch_size, obs_dim)）
        deterministic : bool, optional
            確定的方策を用いるかどうか, by default False

        Returns
        -------
        action : np.ndarray
            行動（shape: (action_dim,) or (batch_size, action_dim)）
        """
        obs_tensor = torch.from_numpy(observation).float().to(self.device)

        with torch.no_grad():
            _, action, _ = self.agent.act(obs_tensor, deterministic)

        return action.cpu().numpy()

    @classmethod
    def from_file(cls, path: str, **kwargs) -> Tuple["AgentController", RunningMeanStd]:
        """
        ファイルからAgentControllerをロードする（Controllerインターフェース実装）

        Parameters
        ----------
        path : str
            保存されたモデルのパス
        **kwargs : dict
            device (torch.device, optional): 使用するデバイス, by default torch.device("cpu")

        Returns
        -------
        controller : AgentController
            ロードされたコントローラー
        obs_rms : RunningMeanStd
            観測値の正規化統計量
        """
        device = kwargs.get("device", torch.device("cpu"))
        state_dict, obs_rms = torch.load(path, map_location=device)
        agent = Agent.from_state_dict(state_dict)
        agent.to(device)
        agent.eval()  # 評価モードに設定
        return cls(agent, device), obs_rms
