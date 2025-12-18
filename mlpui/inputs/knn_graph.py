"""
mlpui/inputs/knn_graph.py

基于 K 近邻的图构建
===================

每个原子固定连接到最近的 K 个邻居
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

from mlpui.core import BaseInputModule, ModelInput


class KNNGraphInput(BaseInputModule):
    """
    基于 K 近邻构建分子图

    对于每个原子，连接到最近的 K 个邻居。
    不使用距离截断，保证每个原子有固定数量的邻居。

    Args:
        k: 每个原子的邻居数
        cutoff: 可选的最大截断距离（超过此距离的邻居会被忽略）
        loop: 是否包含自环边

    Example:
        >>> input_module = KNNGraphInput(k=20)
        >>> model_input = input_module(batch)
    """

    def __init__(
            self,
            k: int = 20,
            cutoff: Optional[float] = None,
            loop: bool = False,
    ):
        super().__init__(cutoff=cutoff or float('inf'), max_neighbors=k, loop=loop)
        self.k = k

    def build_graph(
            self,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        构建 KNN 图
        """
        edge_index = knn_graph(
            pos=pos,
            batch=batch,
            k=self.k,
            loop=self.loop,
        )

        # 计算边向量和距离
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = edge_vec.norm(dim=-1)

        # 可选：过滤超过截断距离的边
        if self.cutoff < float('inf'):
            mask = edge_dist < self.cutoff
            edge_index = edge_index[:, mask]
            edge_vec = edge_vec[mask]
            edge_dist = edge_dist[mask]

        return edge_index, edge_dist, edge_vec, None


def knn_graph(
        pos: Tensor,
        batch: Tensor,
        k: int,
        loop: bool = False,
) -> Tensor:
    """
    构建 K 近邻图

    Args:
        pos: 原子位置 (num_atoms, 3)
        batch: 批次索引 (num_atoms,)
        k: 邻居数
        loop: 是否包含自环

    Returns:
        edge_index: (2, num_edges)
    """
    # 尝试使用 torch_cluster
    try:
        from torch_cluster import knn_graph as _knn_graph
        return _knn_graph(pos, k=k, batch=batch, loop=loop)
    except ImportError:
        pass

    # 纯 PyTorch 实现
    return _knn_graph_pytorch(pos, batch, k, loop)


def _knn_graph_pytorch(
        pos: Tensor,
        batch: Tensor,
        k: int,
        loop: bool = False,
) -> Tensor:
    """
    纯 PyTorch 实现的 knn_graph
    """
    device = pos.device
    num_graphs = int(batch.max()) + 1

    edges_list = []

    for g in range(num_graphs):
        mask = batch == g
        indices = torch.where(mask)[0]
        pos_g = pos[mask]
        num_atoms_g = pos_g.shape[0]

        if num_atoms_g == 0:
            continue

        # 计算距离矩阵
        diff = pos_g.unsqueeze(0) - pos_g.unsqueeze(1)
        dist_matrix = diff.norm(dim=-1)

        # 对于每个原子，找 k 个最近邻
        # 如果不包含自环，先把对角线设为无穷大
        if not loop:
            dist_matrix = dist_matrix + torch.eye(num_atoms_g, device=device) * float('inf')

        # 实际 k 不能超过原子数
        actual_k = min(k, num_atoms_g - (0 if loop else 1))

        if actual_k <= 0:
            continue

        # 找出每行最小的 k 个
        _, neighbors = dist_matrix.topk(actual_k, dim=1, largest=False)

        # 构建边
        row_local = torch.arange(num_atoms_g, device=device).unsqueeze(1).expand(-1, actual_k).flatten()
        col_local = neighbors.flatten()

        # 转换为全局索引
        row_global = indices[row_local]
        col_global = indices[col_local]

        edges_list.append(torch.stack([row_global, col_global], dim=0))

    if len(edges_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.cat(edges_list, dim=1)
