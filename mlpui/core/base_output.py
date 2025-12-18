from abc import ABC, abstractmethod
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core.types import ModelOutput


class BaseOutputHead(nn.Module, ABC):
    """
    输出头的抽象基类

    职责:
      - 接收 ModelOutput (节点特征)
      - 输出目标属性 (energy, dipole, charges, ...)

    每个 OutputHead 只负责一个属性的预测。
    多个 Head 可以共享同一个 ModelOutput。

    子类必须实现:
      - forward(): 计算属性
      - name: 属性名称

    子类可选实现:
      - is_per_atom: 是否是每原子属性
      - required_fields: 需要的 ModelOutput 字段

    Example:
        >>> class EnergyHead(BaseOutputHead):
        ...     @property
        ...     def name(self) -> str:
        ...         return 'energy'
        ...     
        ...     def forward(self, outputs: ModelOutput) -> Tensor:
        ...         atomic_energy = self.mlp(outputs.node_features)
        ...         energy = scatter_sum(atomic_energy, outputs.batch)
        ...         return energy
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        属性名称

        用于:
          - 结果字典的 key
          - 损失函数匹配
          - 日志记录

        Returns:
            str: 如 'energy', 'dipole', 'charges'
        """
        pass

    @property
    def is_per_atom(self) -> bool:
        """
        是否是每原子属性

        - True: 输出形状 (num_atoms, ...) 如 charges, forces
        - False: 输出形状 (num_graphs, ...) 如 energy, dipole

        Returns:
            bool: 默认 False (分子级属性)
        """
        return False

    @property
    def output_dim(self) -> int:
        """
        输出维度

        - energy: 1
        - dipole: 3
        - charges: 1
        - polarizability: 6 (对称矩阵上三角)

        Returns:
            int: 默认 1
        """
        return 1

    @property
    def required_fields(self) -> List[str]:
        """
        需要 ModelOutput 中的哪些字段

        用于验证 ModelOutput 是否包含必要信息。

        Returns:
            List[str]: 默认只需要 node_features

        Example:
            - EnergyHead: ['node_features', 'batch']
            - DipoleHead: ['node_features', 'pos', 'batch']
            - DirectForceHead: ['vector_features', 'batch']
        """
        return ['node_features', 'batch']

    def validate_input(self, outputs: ModelOutput) -> None:
        """
        验证 ModelOutput 是否包含必要字段

        Raises:
            ValueError: 如果缺少必要字段
        """
        for field in self.required_fields:
            value = getattr(outputs, field, None)
            if value is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires '{field}' in ModelOutput, "
                    f"but it is None. Check if the Model provides this field."
                )

    @abstractmethod
    def forward(self, outputs: ModelOutput) -> Tensor:
        """
        计算属性

        Args:
            outputs: 来自 Model 的标准输出 (ModelOutput)

        Returns:
            Tensor: 预测的属性值
                - 分子级属性: (num_graphs, output_dim)
                - 原子级属性: (num_atoms, output_dim)
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"is_per_atom={self.is_per_atom}, "
            f"output_dim={self.output_dim})"
        )


# ============================================================================
# 辅助函数
# ============================================================================

def scatter_sum(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
) -> Tensor:
    """
    按索引求和聚合

    Args:
        src: 源张量
        index: 索引张量
        dim: 聚合维度
        dim_size: 输出维度大小

    Returns:
        聚合后的张量
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    shape = list(src.shape)
    shape[dim] = dim_size

    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)


def scatter_mean(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
) -> Tensor:
    """
    按索引求平均聚合
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    summed = scatter_sum(src, index, dim, dim_size)
    counts = scatter_sum(torch.ones_like(src), index, dim, dim_size)

    return summed / counts.clamp(min=1)