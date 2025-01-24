from typing import Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor

from alg.ppo.distributions import DiagGaussian, FixedNormal
from alg.ppo.utils import init


class Agent(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(Agent, self).__init__()
        self.base = ActorCritic(obs_dim, hidden_dim)
        self.dist = DiagGaussian(hidden_dim, action_dim)

    def forward(self, inputs: Tensor):
        raise NotImplementedError

    def act(self, inputs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        行動を選択する

        Parameters
        ----------
        inputs : Tensor
            観測値
        deterministic : bool, optional
            確定的方策を用いるかどうか, by default False
        """
        value, actor_features = self.base(inputs)
        normal_dist: FixedNormal = self.dist(actor_features)

        if deterministic:
            action = normal_dist.mode()
        else:
            action = normal_dist.sample()

        action_log_probs = normal_dist.log_probs(action)
        # dist_entropy = normal_dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs: Tensor) -> Tensor:
        """状態価値関数の出力を返す"""

        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        行動の評価を行う

        Parameters
        ----------
        inputs : Tensor
            観測値
        action : Tensor
            行動

        Returns
        -------
        value : Tensor
            状態価値関数の出力
        action_log_probs : Tensor
            行動の対数確率
        dist_entropy : Tensor
            分布のエントロピー
        """
        value, actor_features = self.base(inputs)
        normal_dist: FixedNormal = self.dist(actor_features)

        action_log_probs = normal_dist.log_probs(action)
        dist_entropy = normal_dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class ActorCritic(nn.Module):
    """方策と状態価値関数"""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super(ActorCritic, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(), init_(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(), init_(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()
        )
        self.critic_linear = init_(nn.Linear(hidden_dim, 1))

        self.train()

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """状態価値関数の出力と方策の中間層出力を返す"""
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor
