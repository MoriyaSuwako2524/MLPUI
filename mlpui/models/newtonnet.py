"""
mlpui/models/newtonnet.py

NewtonNet 实现
==============

基于牛顿力学的等变消息传递网络

Reference:
    Haghighatlari et al. "NewtonNet: a Newtonian message passing network
    for deep learning of interatomic potentials and forces" (2022)
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core import BaseModel, ModelInput, ModelOutput
from .layers import AtomEmbedding, BesselRBF, PolynomialCutoff, get_activation


class NewtonNet(BaseModel):
    """
    NewtonNet: 牛顿消息传递网络

    特点:
    - 同时维护标量和向量特征
    - 向量特征满足 SE(3) 等变性
    - 基于牛顿第三定律的消息传递

    Args:
        hidden_dim: 隐藏维度
        num_layers: 交互层数
        num_rbf: Bessel 基函数数量
        cutoff: 截断半径
        max_z: 最大原子序数
        activation: 激活函数
        layer_norm: 是否使用层归一化
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            num_layers: int = 3,
            num_rbf: int = 20,
            cutoff: float = 5.0,
            max_z: int = 100,
            activation: str = 'silu',
            layer_norm: bool = False,
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._cutoff = cutoff

        # 原子嵌入
        self.atom_embedding = AtomEmbedding(hidden_dim, max_z)

        # 边嵌入
        self.rbf = BesselRBF(num_rbf, cutoff)
        self.envelope = PolynomialCutoff(cutoff, p=9)

        # 交互层
        act = get_activation(activation)
        self.interactions = nn.ModuleList([
            NewtonNetInteraction(hidden_dim, num_rbf, act, layer_norm)
            for _ in range(num_layers)
        ])

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return True  # NewtonNet 有向量特征

    @property
    def model_name(self) -> str:
        return "NewtonNet"

    def forward(self, inputs: ModelInput) -> ModelOutput:
        z = inputs.z
        edge_index = inputs.edge_index
        edge_dist = inputs.edge_dist
        edge_vec = inputs.edge_vec

        num_atoms = z.shape[0]
        device = z.device
        dtype = inputs.pos.dtype

        # 原子嵌入
        atom_node = self.atom_embedding(z)  # (N, H)

        # 初始化向量特征为零
        force_node = torch.zeros(num_atoms, 3, self._hidden_dim, device=device, dtype=dtype)

        # 边嵌入
        # 归一化距离和方向
        dist_norm = edge_dist / self._cutoff
        dir_edge = edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8)

        # RBF + envelope
        dist_edge = self.envelope(dist_norm) * self.rbf(edge_dist)  # (E, num_rbf)

        # 交互层
        for interaction in self.interactions:
            atom_node, force_node = interaction(
                atom_node, force_node, dir_edge, dist_edge, edge_index
            )

        return ModelOutput.from_input(
            inputs,
            node_features=atom_node,
            vector_features=force_node,
        )


class NewtonNetInteraction(nn.Module):
    """
    NewtonNet 交互层

    同时更新标量特征 (atom_node) 和向量特征 (force_node)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_rbf: int,
            activation: nn.Module,
            layer_norm: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 消息网络 (标量)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_rbf, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 向量消息权重
        self.vec_mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_rbf, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 节点更新
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 向量更新
        self.force_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 可选层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None

    def forward(
            self,
            atom_node: Tensor,  # (N, H) 标量特征
            force_node: Tensor,  # (N, 3, H) 向量特征
            dir_edge: Tensor,  # (E, 3) 边方向
            dist_edge: Tensor,  # (E, num_rbf) RBF 特征
            edge_index: Tensor,  # (2, E)
    ) -> tuple:
        row, col = edge_index
        num_atoms = atom_node.shape[0]

        # === 标量消息 ===
        # 合并源节点特征和边特征
        src_feat = torch.cat([atom_node[col], dist_edge], dim=-1)
        scalar_msg = self.msg_mlp(src_feat)  # (E, H)

        # 聚合
        scalar_agg = torch.zeros_like(atom_node).index_add_(0, row, scalar_msg)

        # === 向量消息 ===
        # 向量权重
        vec_weights = self.vec_mlp(src_feat)  # (E, H)

        # 沿边方向的向量消息
        # msg_vec = dir_edge * weights
        vec_msg = dir_edge.unsqueeze(-1) * vec_weights.unsqueeze(1)  # (E, 3, H)

        # 同时考虑源节点的向量特征
        src_force = force_node[col]  # (E, 3, H)
        vec_msg = vec_msg + src_force * vec_weights.unsqueeze(1)

        # 聚合
        vec_agg = torch.zeros_like(force_node).index_add_(0, row, vec_msg)

        # === 更新 ===
        # 标量更新
        vec_invariant = (force_node ** 2).sum(dim=1)  # (N, H) 向量的模长平方
        node_input = torch.cat([atom_node, scalar_agg + vec_invariant], dim=-1)
        atom_node_new = atom_node + self.node_mlp(node_input)

        # 向量更新
        force_input = torch.cat([atom_node.unsqueeze(1).expand(-1, 3, -1), vec_agg], dim=-1)
        force_node_new = force_node + self.force_mlp(force_input) * force_node

        # 层归一化
        if self.layer_norm is not None:
            atom_node_new = self.layer_norm(atom_node_new)

        return atom_node_new, force_node_new