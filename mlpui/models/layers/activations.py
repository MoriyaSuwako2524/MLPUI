"""
mlpui/models/layers/activations.py

激活函数
========
"""

import torch
import torch.nn as nn
from torch import Tensor


class ShiftedSoftplus(nn.Module):
    """
    移位 Softplus (SchNet 使用)

    f(x) = softplus(x) - log(2)
    """

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.softplus(x) - self.shift


class Swish(nn.Module):
    """
    Swish / SiLU 激活

    f(x) = x * sigmoid(x)
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


# 注册表
ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'swish': nn.SiLU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'softplus': nn.Softplus,
    'shifted_softplus': ShiftedSoftplus,
}


def get_activation(name: str) -> nn.Module:
    """根据名称获取激活函数"""
    if name.lower() not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATION_REGISTRY.keys())}")
    return ACTIVATION_REGISTRY[name.lower()]()