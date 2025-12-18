"""
mlpui/models/layers/radial.py

径向基函数
==========

将距离标量扩展为特征向量
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class GaussianRBF(nn.Module):
    """
    高斯径向基函数

    rbf(r) = exp(-gamma * (r - center)^2)

    Args:
        num_rbf: 基函数数量
        cutoff: 截断半径
        trainable: 是否可训练
    """

    def __init__(
            self,
            num_rbf: int = 50,
            cutoff: float = 5.0,
            trainable: bool = False,
    ):
        super().__init__()

        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # 均匀分布的中心点
        centers = torch.linspace(0, cutoff, num_rbf)
        width = (centers[1] - centers[0]) if num_rbf > 1 else cutoff
        gamma = 1.0 / (width ** 2)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: (num_edges,) 距离
        Returns:
            (num_edges, num_rbf) RBF 特征
        """
        dist = dist.unsqueeze(-1)  # (num_edges, 1)
        return torch.exp(-self.gamma * (dist - self.centers) ** 2)

    def reset_parameters(self):
        pass


class BesselRBF(nn.Module):
    """
    Bessel 径向基函数 (DimeNet 使用)

    rbf_n(r) = sqrt(2/c) * sin(n*pi*r/c) / r

    Args:
        num_rbf: 基函数数量
        cutoff: 截断半径
        trainable: 是否可训练频率
    """

    def __init__(
            self,
            num_rbf: int = 20,
            cutoff: float = 5.0,
            trainable: bool = False,
    ):
        super().__init__()

        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # 频率 n*pi/cutoff
        freqs = torch.arange(1, num_rbf + 1) * math.pi / cutoff

        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

        # 归一化因子
        self.register_buffer('norm', torch.tensor(math.sqrt(2.0 / cutoff)))

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: (num_edges,) 距离
        Returns:
            (num_edges, num_rbf) RBF 特征
        """
        dist = dist.unsqueeze(-1)  # (num_edges, 1)
        # 避免除零
        dist = dist.clamp(min=1e-8)
        return self.norm * torch.sin(self.freqs * dist) / dist

    def reset_parameters(self):
        pass


class ExpNormRBF(nn.Module):
    """
    指数归一化径向基函数 (TorchMD-Net 使用)

    Args:
        num_rbf: 基函数数量
        cutoff_lower: 下截断
        cutoff_upper: 上截断
        trainable: 是否可训练
    """

    def __init__(
            self,
            num_rbf: int = 50,
            cutoff_lower: float = 0.0,
            cutoff_upper: float = 5.0,
            trainable: bool = False,
    ):
        super().__init__()

        self.num_rbf = num_rbf
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        # 均匀分布的中心点 (指数空间)
        alpha = 5.0 / (cutoff_upper - cutoff_lower)
        means = torch.linspace(
            math.exp(-cutoff_upper * alpha),
            math.exp(-cutoff_lower * alpha),
            num_rbf,
        )
        betas = torch.full((num_rbf,), (2 / num_rbf * (1 - math.exp(-cutoff_upper * alpha))) ** -2)

        if trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

        self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        return torch.exp(-self.betas * (torch.exp(-self.alpha * dist) - self.means) ** 2)

    def reset_parameters(self):
        pass


# 注册表
RBF_REGISTRY = {
    'gaussian': GaussianRBF,
    'bessel': BesselRBF,
    'expnorm': ExpNormRBF,
}


def get_rbf(name: str, **kwargs) -> nn.Module:
    """根据名称获取 RBF"""
    if name not in RBF_REGISTRY:
        raise ValueError(f"Unknown RBF: {name}. Available: {list(RBF_REGISTRY.keys())}")
    return RBF_REGISTRY[name](**kwargs)