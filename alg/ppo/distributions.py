import torch
import torch.nn as nn
from torch import Tensor

from alg.ppo.ppo_utils import AddBias, init


class FixedNormal(torch.distributions.Normal):
    """等方正規分布を仮定した`torch.distributions.Normal`の拡張クラス"""

    def log_probs(self, actions: Tensor) -> Tensor:
        """行動の対数尤度を計算する"""
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        """エントロピーを計算する"""
        return super().entropy().sum(-1)

    def mode(self):
        """最頻値を取得する"""
        return self.mean


class DiagGaussian(nn.Module):
    """等方正規分布"""

    def __init__(self, num_inputs: int, num_outputs: int):
        super(DiagGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x: Tensor) -> FixedNormal:
        mean = self.fc_mean(x)

        zeros = torch.zeros(mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        logstd: Tensor = self.logstd(zeros)
        return FixedNormal(mean, logstd.exp())
