"""
SchNet
Reference:
    Schütt et al. "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions" (NeurIPS 2017)
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core import BaseModel, ModelInput, ModelOutput
from .layers import AtomEmbedding, GaussianRBF, CosineCutoff, get_activation


class SchNet(BaseModel):
    """
    SchNet: 连续滤波器卷积神经网络

    特点:
    - 简单直观，易于理解
    - 不变模型 (无向量特征)
    - 适合小分子

    Args:
        hidden_dim: 隐藏层维度
        num_layers: 交互层数量
        num_rbf: RBF 数量
        cutoff: 截断半径
        max_z: 最大原子序数
        activation: 激活函数

    Example:
        >>> model = SchNet(hidden_dim=128, num_layers=6)
        >>> output = model(model_input)
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            num_layers: int = 6,
            num_rbf: int = 50,
            cutoff: float = 5.0,
            max_z: int = 100,
            activation: str = 'shifted_softplus',
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._cutoff = cutoff

        # 原子嵌入
        self.embedding = AtomEmbedding(hidden_dim, max_z)

        # RBF
        self.rbf = GaussianRBF(num_rbf, cutoff)

        # 交互层
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_rbf, cutoff, activation)
            for _ in range(num_layers)
        ])

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "SchNet"

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """
        前向传播

        Args:
            inputs: ModelInput
        Returns:
            ModelOutput
        """
        z = inputs.z
        edge_index = inputs.edge_index
        edge_dist = inputs.edge_dist

        # 原子嵌入
        x = self.embedding(z)  # (num_atoms, hidden_dim)

        # RBF 扩展
        edge_attr = self.rbf(edge_dist)  # (num_edges, num_rbf)

        # 交互层
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_dist, edge_attr)

        return ModelOutput.from_input(
            inputs,
            node_features=x,
            vector_features=None,
        )


class SchNetInteraction(nn.Module):
    """
    SchNet 交互层

    x_i' = x_i + Σ_j x_j * W(r_ij)

    其中 W 是连续滤波器
    """

    def __init__(
            self,
            hidden_dim: int,
            num_rbf: int,
            cutoff: float,
            activation: str = 'shifted_softplus',
    ):
        super().__init__()

        act = get_activation(activation)

        # 连续滤波器生成网络
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 截断
        self.cutoff = CosineCutoff(cutoff)

        # 原子特征转换
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_dist: Tensor,
            edge_attr: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (num_atoms, hidden_dim) 节点特征
            edge_index: (2, num_edges)
            edge_dist: (num_edges,)
            edge_attr: (num_edges, num_rbf) RBF 特征
        """
        row, col = edge_index

        # 生成滤波器权重
        W = self.filter_net(edge_attr)  # (num_edges, hidden_dim)

        # 应用截断
        C = self.cutoff(edge_dist).unsqueeze(-1)  # (num_edges, 1)
        W = W * C

        # 消息传递: msg = x[col] * W
        msg = x[col] * W  # (num_edges, hidden_dim)

        # 聚合
        x_agg = torch.zeros_like(x).index_add_(0, row, msg)

        # 更新
        x_out = self.atom_net(x_agg)

        return x_out