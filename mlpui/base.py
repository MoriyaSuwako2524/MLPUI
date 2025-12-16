
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class MLPOutput:
    """
    统一的模型输出容器

    所有模型（NewtonNet, TorchMD-Net, MACE 等）的表示层输出都转换为这个格式，
    使得输出头可以统一处理。

    Attributes:
        node_features: 节点特征 (num_atoms, hidden_dim)
        z: 原子序数 (num_atoms,)
        pos: 原子位置 (num_atoms, 3)
        batch: 批次索引 (num_atoms,)
        vector_features: 可选的等变向量特征
        edge_index: 可选的边索引
        cell: 可选的晶胞矩阵
        properties: 存储计算的属性
    """
    node_features: Tensor
    z: Tensor
    pos: Tensor
    batch: Tensor
    vector_features: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    edge_attr: Optional[Tensor] = None
    cell: Optional[Tensor] = None
    displacement: Optional[Tensor] = None
    properties: Dict[str, Tensor] = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        return self.z.shape[0]

    @property
    def num_graphs(self) -> int:
        return int(self.batch.max()) + 1

    @property
    def device(self) -> torch.device:
        return self.z.device


class BaseMLPWrapper(nn.Module, ABC):
    """
    MLP 表示模型的抽象基类

    每个具体的模型（NewtonNet, TorchMD-Net, MACE）都需要实现一个继承此类的 Wrapper，
    将各自的表示层输出转换为统一的 MLPOutput 格式。

    子类必须实现:
        - forward(): 提取节点特征
        - hidden_dim: 返回特征维度
    """

    @abstractmethod
    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """
        前向传播，提取节点表示

        Args:
            z: 原子序数 (num_atoms,)
            pos: 原子位置 (num_atoms, 3)
            batch: 批次索引 (num_atoms,)
            cell: 晶胞矩阵 (num_graphs, 3, 3) 或 None

        Returns:
            MLPOutput 包含节点特征和元数据
        """
        pass

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """返回节点特征的维度"""
        pass

    @property
    def has_vector_features(self) -> bool:
        """是否输出等变向量特征"""
        return False

    @property
    def model_name(self) -> str:
        """模型名称"""
        return self.__class__.__name__


class BaseOutputHead(nn.Module, ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """属性名称（如 'energy', 'dipole'）"""
        pass

    @property
    def is_per_atom(self) -> bool:
        """是否是每原子属性"""
        return False

    @property
    def requires_grad(self) -> bool:
        """是否需要梯度计算"""
        return False

    @abstractmethod
    def forward(self, mlp_output: MLPOutput) -> Tensor:
        """
        计算属性

        Args:
            mlp_output: 统一的模型输出

        Returns:
            计算的属性张量
        """
        pass


# =============================================================================
# 工具函数
# =============================================================================

def scatter_sum(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None
) -> Tensor:
    """
    Scatter sum 操作，将原子级别的值聚合到分子级别

    Args:
        src: 源张量
        index: 索引张量（表示每个原子属于哪个分子）
        dim: 聚合维度
        dim_size: 输出维度大小

    Returns:
        聚合后的张量
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    size = list(src.size())
    size[dim] = dim_size

    out = torch.zeros(size, dtype=src.dtype, device=src.device)

    if src.dim() > 1:
        index_expanded = index.view(*([1] * dim + [-1] + [1] * (src.dim() - dim - 1)))
        index_expanded = index_expanded.expand_as(src)
    else:
        index_expanded = index

    return out.scatter_add_(dim, index_expanded, src)


def scatter_mean(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None
) -> Tensor:
    """Scatter mean 操作"""
    sum_result = scatter_sum(src, index, dim, dim_size)
    count = scatter_sum(torch.ones_like(src), index, dim, dim_size)
    return sum_result / count.clamp(min=1)