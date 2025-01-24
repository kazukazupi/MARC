import torch.nn as nn
from torch import Tensor


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias: Tensor):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2
        bias = self._bias.t().view(1, -1)  # (1, N)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):

    lr = initial_lr - initial_lr * (epoch / total_num_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
