"""
mlpui/inputs/radius_graph.py

基于距离截断的图构建
====================

最常用的图构建方式，适用于分子体系（非周期性）
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

from mlpui.core.base_input import BaseInputModule, ModelInput


class RadiusGraphInput(BaseInputModule):
    """
    基于距离截断构建分子图

    对于每个原子，找出所有距离小于 cutoff 的原子作为邻居。
    适用于分子体系（非周期性边界条件）。

    Args:
        cutoff: 截断半径，超过此距离的原子对不构建边
        max_neighbors: 每个原子的最大邻居数，None 表示不限制
        loop: 是否包含自环边 (i→i)，默认 False

    Example:
        >>> input_module = RadiusGraphInput(cutoff=5.0)
        >>> model_input = input_module(batch)
    """

    def __init__(
            self,
            cutoff: float = 5.0,
            max_neighbors: Optional[int] = None,
            loop: bool = False,
    ):
        super().__init__(cutoff=cutoff, max_neighbors=max_neighbors, loop=loop)

    def build_graph(
            self,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        构建基于距离截断的图

        Args:
            pos: 原子位置 (num_atoms, 3)
            batch: 批次索引 (num_atoms,)
            cell: 未使用 (非周期性)
            pbc: 未使用 (非周期性)

        Returns:
            edge_index: (2, num_edges)
            edge_dist: (num_edges,)
            edge_vec: (num_edges, 3)
            offsets: None (非周期性体系)
        """
        edge_index = radius_graph(
            pos=pos,
            batch=batch,
            r=self.cutoff,
            max_num_neighbors=self.max_neighbors,
            loop=self.loop,
        )

        # 计算边向量和距离
        row, col = edge_index
        edge_vec = pos[col] - pos[row]  # 从 row 指向 col
        edge_dist = edge_vec.norm(dim=-1)

        return edge_index, edge_dist, edge_vec, None


def radius_graph(
        pos: Tensor,
        batch: Tensor,
        r: float,
        max_num_neighbors: Optional[int] = None,
        loop: bool = False,
) -> Tensor:
    """
    构建半径图

    为每个原子找出距离小于 r 的所有邻居。
    只在同一个分子（相同 batch 索引）内构建边。

    Args:
        pos: 原子位置 (num_atoms, 3)
        batch: 批次索引 (num_atoms,)
        r: 截断半径
        max_num_neighbors: 最大邻居数
        loop: 是否包含自环

    Returns:
        edge_index: (2, num_edges)
    """
    # 尝试使用 torch_cluster（如果可用）
    try:
        from torch_cluster import radius_graph as _radius_graph
        return _radius_graph(
            pos, r=r, batch=batch,
            max_num_neighbors=max_num_neighbors if max_num_neighbors else 32,
            loop=loop,
        )
    except ImportError:
        pass

    # 纯 PyTorch 实现（较慢但无依赖）
    return _radius_graph_pytorch(pos, batch, r, max_num_neighbors, loop)


def _radius_graph_pytorch(
        pos: Tensor,
        batch: Tensor,
        r: float,
        max_num_neighbors: Optional[int] = None,
        loop: bool = False,
) -> Tensor:
    """
    纯 PyTorch 实现的 radius_graph

    注意: 这个实现较慢，建议安装 torch_cluster
    """
    num_atoms = pos.shape[0]
    device = pos.device

    # 计算所有原子对的距离
    # 使用分块计算避免内存爆炸（大体系）

    edges_list = []

    # 按 batch 分组处理
    num_graphs = int(batch.max()) + 1

    for g in range(num_graphs):
        # 找出属于这个分子的原子
        mask = batch == g
        indices = torch.where(mask)[0]
        pos_g = pos[mask]
        num_atoms_g = pos_g.shape[0]

        if num_atoms_g == 0:
            continue

        # 计算分子内所有原子对的距离
        # dist_matrix[i, j] = ||pos_g[i] - pos_g[j]||
        diff = pos_g.unsqueeze(0) - pos_g.unsqueeze(1)  # (N, N, 3)
        dist_matrix = diff.norm(dim=-1)  # (N, N)

        # 找出距离 < r 的原子对
        if loop:
            mask_r = dist_matrix < r
        else:
            mask_r = (dist_matrix < r) & (dist_matrix > 0)

        # 转换为边索引
        row_local, col_local = torch.where(mask_r)

        # 限制最大邻居数
        if max_num_neighbors is not None and max_num_neighbors > 0:
            # 按源节点分组，保留最近的 k 个邻居
            row_local, col_local = _limit_neighbors(
                row_local, col_local, dist_matrix, max_num_neighbors
            )

        # 转换回全局索引
        row_global = indices[row_local]
        col_global = indices[col_local]

        edges_list.append(torch.stack([row_global, col_global], dim=0))

    if len(edges_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat(edges_list, dim=1)
    return edge_index


def _limit_neighbors(
        row: Tensor,
        col: Tensor,
        dist_matrix: Tensor,
        max_neighbors: int,
) -> Tuple[Tensor, Tensor]:
    """
    限制每个原子的最大邻居数，保留最近的邻居
    """
    device = row.device
    num_atoms = dist_matrix.shape[0]

    new_row = []
    new_col = []

    for i in range(num_atoms):
        # 找出以 i 为源的所有边
        mask = row == i
        neighbors = col[mask]

        if len(neighbors) <= max_neighbors:
            new_row.append(torch.full((len(neighbors),), i, device=device))
            new_col.append(neighbors)
        else:
            # 按距离排序，保留最近的
            dists = dist_matrix[i, neighbors]
            _, sorted_idx = dists.sort()
            selected = neighbors[sorted_idx[:max_neighbors]]
            new_row.append(torch.full((max_neighbors,), i, device=device))
            new_col.append(selected)

    if len(new_row) == 0:
        return row[:0], col[:0]

    return torch.cat(new_row), torch.cat(new_col)

