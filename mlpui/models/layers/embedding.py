"""
mlpui/models/layers/embedding.py

原子嵌入层
==========

将原子序数转换为特征向量
"""

import torch
import torch.nn as nn
from torch import Tensor


class AtomEmbedding(nn.Module):
    """
    原子序数嵌入

    Args:
        hidden_dim: 嵌入维度
        max_z: 最大原子序数
    """

    def __init__(self, hidden_dim: int, max_z: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(max_z + 1, hidden_dim, padding_idx=0)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (num_atoms,) 原子序数
        Returns:
            (num_atoms, hidden_dim) 原子嵌入
        """
        return self.embedding(z)

    def reset_parameters(self):
        self.embedding.reset_parameters()


class NodeEmbedding(nn.Module):
    """
    带邻居信息的节点嵌入

    结合原子类型和邻居类型信息

    Args:
        hidden_dim: 嵌入维度
        num_rbf: RBF 维度
        cutoff: 截断半径
        max_z: 最大原子序数
    """

    def __init__(
            self,
            hidden_dim: int,
            num_rbf: int,
            cutoff: float,
            max_z: int = 100,
    ):
        super().__init__()

        self.atom_embedding = nn.Embedding(max_z + 1, hidden_dim, padding_idx=0)
        self.neighbor_embedding = nn.Embedding(max_z + 1, hidden_dim, padding_idx=0)
        self.distance_proj = nn.Linear(num_rbf, hidden_dim)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

        from .cutoff import CosineCutoff
        self.cutoff = CosineCutoff(cutoff)

    def forward(
            self,
            z: Tensor,
            edge_index: Tensor,
            edge_dist: Tensor,
            edge_attr: Tensor,  # RBF 特征
    ) -> Tensor:
        """
        Args:
            z: (num_atoms,) 原子序数
            edge_index: (2, num_edges)
            edge_dist: (num_edges,) 距离
            edge_attr: (num_edges, num_rbf) RBF 特征
        """
        row, col = edge_index

        # 原子嵌入
        x = self.atom_embedding(z)

        # 邻居嵌入
        C = self.cutoff(edge_dist)
        W = self.distance_proj(edge_attr) * C.unsqueeze(-1)
        neighbor_feat = self.neighbor_embedding(z)

        # 聚合邻居信息
        msg = W * neighbor_feat[col]
        neighbor_agg = torch.zeros_like(x).index_add_(0, row, msg)

        # 合并
        x = self.combine(torch.cat([x, neighbor_agg], dim=-1))

        return x