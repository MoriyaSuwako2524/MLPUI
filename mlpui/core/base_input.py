

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core.types import ModelInput


class BaseInputModule(nn.Module, ABC):
    """
    InputModule 抽象基类

    职责:
      1. 从 DataLoader 的 batch 中提取原子信息
      2. 构建分子图 (邻居关系)
      3. 计算边特征 (距离、方向向量)
      4. 输出标准的 ModelInput

    子类需要实现:
      - build_graph(): 构建图结构的具体逻辑

    Example:
        >>> input_module = RadiusGraphInput(cutoff=5.0)
        >>> model_input = input_module(batch)
        >>> # model_input 是标准的 ModelInput 对象
    """

    def __init__(
            self,
            cutoff: float = 5.0,
            max_neighbors: Optional[int] = None,
            loop: bool = False,
    ):
        """
        Args:
            cutoff: 截断半径，超过此距离的原子对不构建边
            max_neighbors: 每个原子的最大邻居数，None 表示不限制
            loop: 是否包含自环边 (i→i)
        """
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.loop = loop

    @abstractmethod
    def build_graph(
            self,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None,
    ) -> tuple:
        """
        构建分子图

        Args:
            pos: 原子位置 (num_atoms, 3)
            batch: 批次索引 (num_atoms,)
            cell: 晶胞矩阵 (num_graphs, 3, 3)
            pbc: 周期性边界条件 (3,)

        Returns:
            edge_index: (2, num_edges)
            edge_dist: (num_edges,)
            edge_vec: (num_edges, 3)
            offsets: (num_edges, 3) 或 None，周期性偏移
        """
        pass

    def extract_from_batch(self, batch: Any) -> dict:
        """
        从 DataLoader 的 batch 中提取数据

        支持多种输入格式:
          - PyG Data/Batch 对象
          - 字典
          - 自定义对象 (需要有 z, pos, batch 属性)

        Returns:
            dict: 包含 z, pos, batch, cell, pbc 等字段
        """
        # PyG Batch 对象
        if hasattr(batch, 'z') and hasattr(batch, 'pos'):
            return {
                'z': batch.z,
                'pos': batch.pos,
                'batch': getattr(batch, 'batch',
                                 torch.zeros(batch.z.shape[0], dtype=torch.long, device=batch.z.device)),
                'cell': getattr(batch, 'cell', None),
                'pbc': getattr(batch, 'pbc', None),
                'num_graphs': int(batch.batch.max().item() + 1) if hasattr(batch, 'batch') else 1,
            }

        # 字典格式
        if isinstance(batch, dict):
            z = batch['z']
            return {
                'z': z,
                'pos': batch['pos'],
                'batch': batch.get('batch', torch.zeros(z.shape[0], dtype=torch.long, device=z.device)),
                'cell': batch.get('cell', None),
                'pbc': batch.get('pbc', None),
                'num_graphs': batch.get('num_graphs', 1),
            }

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def forward(self, batch: Any) -> ModelInput:
        """
        将原始 batch 转换为 ModelInput

        Args:
            batch: 来自 DataLoader 的批次数据

        Returns:
            ModelInput: 标准化的模型输入
        """
        # 1. 提取原始数据
        data = self.extract_from_batch(batch)

        z = data['z']
        pos = data['pos']
        batch_idx = data['batch']
        cell = data['cell']
        pbc = data['pbc']
        num_graphs = data['num_graphs']

        # 2. 构建图
        edge_index, edge_dist, edge_vec, offsets = self.build_graph(
            pos=pos,
            batch=batch_idx,
            cell=cell,
            pbc=pbc,
        )

        # 3. 构建 ModelInput
        return ModelInput(
            z=z,
            pos=pos,
            batch=batch_idx,
            edge_index=edge_index,
            edge_dist=edge_dist,
            edge_vec=edge_vec,
            cell=cell,
            pbc=pbc,
            offsets=offsets,
            num_graphs=num_graphs,
        )

