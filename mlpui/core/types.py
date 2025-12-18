from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from torch import Tensor


@dataclass
class ModelInput:
    """
    Model 的标准输入格式

    由 InputModule 构建，包含原子信息和图结构。
    所有 Model 都接收这个统一格式。

    Attributes:
        z: 原子序数 (num_atoms,)
        pos: 原子位置 (num_atoms, 3)
        batch: 批次索引 (num_atoms,)，指示每个原子属于哪个分子

        edge_index: 边索引 (2, num_edges)，[源节点, 目标节点]
        edge_dist: 边距离 (num_edges,)
        edge_vec: 边向量 (num_edges, 3)，从源到目标的方向

        cell: 晶胞矩阵 (num_graphs, 3, 3)，周期性体系使用
        pbc: 周期性边界条件 (3,)，[x, y, z] 方向是否周期

        num_graphs: 批次中分子数量
    """

    # === 原子信息 (必需) ===
    z: Tensor  # (num_atoms,)
    pos: Tensor  # (num_atoms, 3)
    batch: Tensor  # (num_atoms,)

    # === 图结构 (必需，由 InputModule 构建) ===
    edge_index: Tensor  # (2, num_edges)
    edge_dist: Tensor  # (num_edges,)
    edge_vec: Tensor  # (num_edges, 3)

    # === 周期性体系 (可选) ===
    cell: Optional[Tensor] = None  # (num_graphs, 3, 3)
    pbc: Optional[Tensor] = None  # (3,)
    offsets: Optional[Tensor] = None  # (num_edges, 3) 周期性偏移

    # === 元信息 ===
    num_graphs: int = 1

    # === 扩展字段 (存储额外信息) ===
    extra: Dict[str, Any] = field(default_factory=dict)

    # ----- 便捷属性 -----

    @property
    def num_atoms(self) -> int:
        """总原子数"""
        return self.z.shape[0]

    @property
    def num_edges(self) -> int:
        """总边数"""
        return self.edge_index.shape[1]

    @property
    def device(self) -> torch.device:
        """数据所在设备"""
        return self.z.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pos.dtype

    def to(self, device: torch.device) -> 'ModelInput':
        return ModelInput(
            z=self.z.to(device),
            pos=self.pos.to(device),
            batch=self.batch.to(device),
            edge_index=self.edge_index.to(device),
            edge_dist=self.edge_dist.to(device),
            edge_vec=self.edge_vec.to(device),
            cell=self.cell.to(device) if self.cell is not None else None,
            pbc=self.pbc.to(device) if self.pbc is not None else None,
            offsets=self.offsets.to(device) if self.offsets is not None else None,
            num_graphs=self.num_graphs,
            extra={k: v.to(device) if isinstance(v, Tensor) else v
                   for k, v in self.extra.items()},
        )


@dataclass
class ModelOutput:
    """
    Model 的标准输出格式

    由 Model 产生，包含节点特征。
    所有 OutputHead 都接收这个统一格式。

    Attributes:
        node_features: 节点标量特征 (num_atoms, hidden_dim)
        vector_features: 节点向量特征 (num_atoms, 3, hidden_dim)，等变模型输出

        (以下为透传字段，从 ModelInput 传递，供 OutputHead 使用)
        z, pos, batch, edge_index, ...
    """

    # === 核心输出 (必需) ===
    node_features: Tensor  # (num_atoms, hidden_dim)

    # === 等变向量特征 (可选，NewtonNet/MACE/PaiNN 等输出) ===
    vector_features: Optional[Tensor] = None  # (num_atoms, 3, hidden_dim)

    # === 透传字段 (从 ModelInput 传递) ===
    z: Optional[Tensor] = None  # (num_atoms,)
    pos: Optional[Tensor] = None  # (num_atoms, 3)
    batch: Optional[Tensor] = None  # (num_atoms,)
    edge_index: Optional[Tensor] = None  # (2, num_edges)
    edge_dist: Optional[Tensor] = None  # (num_edges,)
    edge_vec: Optional[Tensor] = None  # (num_edges, 3)
    cell: Optional[Tensor] = None  # (num_graphs, 3, 3)

    # === 元信息 ===
    num_graphs: int = 1
    hidden_dim: int = 0

    # === 扩展字段 ===
    extra: Dict[str, Any] = field(default_factory=dict)


    @property
    def num_atoms(self) -> int:
        return self.node_features.shape[0]

    @property
    def has_vector_features(self) -> bool:
        return self.vector_features is not None

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @classmethod
    def from_input(
            cls,
            model_input: ModelInput,
            node_features: Tensor,
            vector_features: Optional[Tensor] = None,
            **kwargs
    ) -> 'ModelOutput':
        """
        从 ModelInput 创建 ModelOutput，自动透传必要字段

        这是推荐的创建方式，确保信息不丢失
        """
        return cls(
            node_features=node_features,
            vector_features=vector_features,
            z=model_input.z,
            pos=model_input.pos,
            batch=model_input.batch,
            edge_index=model_input.edge_index,
            edge_dist=model_input.edge_dist,
            edge_vec=model_input.edge_vec,
            cell=model_input.cell,
            num_graphs=model_input.num_graphs,
            hidden_dim=node_features.shape[-1],
            **kwargs
        )
