"""
mlpui/models/layers/cutoff.py

截断函数
========

平滑截断边权重
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class CosineCutoff(nn.Module):
    """
    余弦截断函数

    f(r) = 0.5 * (cos(pi * r / cutoff) + 1)  if r < cutoff
         = 0                                  otherwise
    """

    def __init__(self, cutoff: float, cutoff_lower: float = 0.0):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_lower = cutoff_lower

    def forward(self, dist: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            # 双边截断
            cutoffs = 0.5 * (
                    torch.cos(math.pi * (2 * (dist - self.cutoff_lower) / (self.cutoff - self.cutoff_lower) + 1)) + 1
            )
            cutoffs = cutoffs * (dist > self.cutoff_lower).float()
            cutoffs = cutoffs * (dist < self.cutoff).float()
        else:
            # 单边截断
            cutoffs = 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1)
            cutoffs = cutoffs * (dist < self.cutoff).float()

        return cutoffs


class PolynomialCutoff(nn.Module):
    """
    多项式截断函数 (更平滑)

    f(r) = 1 - (p+1)*(p+2)/2 * x^p + p*(p+2) * x^(p+1) - p*(p+1)/2 * x^(p+2)
    where x = r / cutoff
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, dist: Tensor) -> Tensor:
        x = dist / self.cutoff
        p = self.p

        cutoffs = 1 - (p + 1) * (p + 2) / 2 * x.pow(p) \
                  + p * (p + 2) * x.pow(p + 1) \
                  - p * (p + 1) / 2 * x.pow(p + 2)

        cutoffs = cutoffs * (dist < self.cutoff).float()
        return cutoffs.clamp(min=0, max=1)


class MollifierCutoff(nn.Module):
    """
    Mollifier 截断函数 (无限可微)
    """

    def __init__(self, cutoff: float, eps: float = 1e-8):
        super().__init__()
        self.cutoff = cutoff
        self.eps = eps

    def forward(self, dist: Tensor) -> Tensor:
        x = dist / self.cutoff
        mask = x < 1

        cutoffs = torch.zeros_like(dist)
        x_valid = x[mask]
        cutoffs[mask] = torch.exp(-1.0 / (1 - x_valid.pow(2) + self.eps))

        return cutoffs