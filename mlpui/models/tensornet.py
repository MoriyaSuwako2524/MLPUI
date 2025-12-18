"""
mlpui/models/tensornet.py

TensorNet 实现
==============

基于笛卡尔张量的等变/不变表示

Reference:
    Simeon & de Fabritiis. "TensorNet: Cartesian Tensor Representations
    for Efficient Learning of Molecular Potentials" (NeurIPS 2023)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core.base_model import BaseModel, ModelInput, ModelOutput
from .layers import AtomEmbedding, ExpNormRBF, CosineCutoff, get_activation


class TensorNet(BaseModel):
    """
    TensorNet: 笛卡尔张量表示网络

    特点:
    - 使用 3x3 张量表示 (I, A, S 分解)
    - O(3) 或 SO(3) 等变
    - 高效计算

    Args:
        hidden_dim: 隐藏维度
        num_layers: 交互层数
        num_rbf: RBF 数量
        cutoff: 截断半径
        max_z: 最大原子序数
        activation: 激活函数
        equivariance_group: 等变群 ('O(3)' 或 'SO(3)')
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            num_layers: int = 2,
            num_rbf: int = 32,
            cutoff: float = 5.0,
            max_z: int = 100,
            activation: str = 'silu',
            equivariance_group: str = 'O(3)',
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._cutoff = cutoff
        self.equivariance_group = equivariance_group

        act_class = get_activation(activation).__class__

        # RBF
        self.rbf = ExpNormRBF(num_rbf, cutoff_upper=cutoff)

        # 张量嵌入
        self.tensor_embedding = TensorEmbedding(
            hidden_dim, num_rbf, cutoff, max_z, act_class
        )

        # 交互层
        self.interactions = nn.ModuleList([
            TensorInteraction(hidden_dim, num_rbf, cutoff, equivariance_group, act_class)
            for _ in range(num_layers)
        ])

        # 输出层
        self.out_norm = nn.LayerNorm(3 * hidden_dim)
        self.out_linear = nn.Linear(3 * hidden_dim, hidden_dim)
        self.act = act_class()

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return False  # TensorNet 输出标量特征

    @property
    def model_name(self) -> str:
        return "TensorNet"

    def forward(self, inputs: ModelInput) -> ModelOutput:
        z = inputs.z
        edge_index = inputs.edge_index
        edge_dist = inputs.edge_dist
        edge_vec = inputs.edge_vec

        # RBF
        edge_attr = self.rbf(edge_dist)

        # 归一化边向量
        edge_vec_norm = edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8)

        # 张量嵌入
        X = self.tensor_embedding(z, edge_index, edge_dist, edge_vec_norm, edge_attr)

        # 交互层
        for interaction in self.interactions:
            X = interaction(X, edge_index, edge_dist, edge_attr)

        # 输出: 从张量提取标量特征
        I, A, S = decompose_tensor(X)
        x = torch.cat([tensor_norm(I), tensor_norm(A), tensor_norm(S)], dim=-1)
        x = self.out_norm(x)
        x = self.act(self.out_linear(x))

        return ModelOutput.from_input(inputs, node_features=x, vector_features=None)


class TensorEmbedding(nn.Module):
    """张量嵌入层"""

    def __init__(
            self,
            hidden_dim: int,
            num_rbf: int,
            cutoff: float,
            max_z: int,
            act_class,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 原子嵌入
        self.atom_emb = nn.Embedding(max_z + 1, hidden_dim, padding_idx=0)
        self.atom_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # 距离投影 (I, A, S 三个分量)
        self.dist_proj_I = nn.Linear(num_rbf, hidden_dim)
        self.dist_proj_A = nn.Linear(num_rbf, hidden_dim)
        self.dist_proj_S = nn.Linear(num_rbf, hidden_dim)

        # 截断
        self.cutoff = CosineCutoff(cutoff)

        # 张量线性层
        self.linear_I = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_S = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 标量 MLP
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            act_class(),
            nn.Linear(2 * hidden_dim, 3 * hidden_dim),
        )

    def forward(
            self,
            z: Tensor,
            edge_index: Tensor,
            edge_dist: Tensor,
            edge_vec_norm: Tensor,
            edge_attr: Tensor,
    ) -> Tensor:
        """返回 (num_atoms, hidden_dim, 3, 3) 张量"""

        row, col = edge_index

        # 原子对嵌入
        z_emb = self.atom_emb(z)
        z_pair = torch.cat([z_emb[row], z_emb[col]], dim=-1)
        Zij = self.atom_proj(z_pair).unsqueeze(-1).unsqueeze(-1)  # (E, H, 1, 1)

        # 截断
        C = self.cutoff(edge_dist).view(-1, 1, 1, 1) * Zij

        # 构建 I, A, S 张量
        eye = torch.eye(3, device=edge_vec_norm.device).unsqueeze(0).unsqueeze(0)

        Iij = self.dist_proj_I(edge_attr).unsqueeze(-1).unsqueeze(-1) * C * eye
        Aij = self.dist_proj_A(edge_attr).unsqueeze(-1).unsqueeze(-1) * C * vector_to_skew(edge_vec_norm)
        Sij = self.dist_proj_S(edge_attr).unsqueeze(-1).unsqueeze(-1) * C * vector_to_sym(edge_vec_norm)

        # 聚合
        num_atoms = z.shape[0]
        shape = (num_atoms, self.hidden_dim, 3, 3)

        I = torch.zeros(shape, device=z.device, dtype=Iij.dtype).index_add_(0, row, Iij)
        A = torch.zeros(shape, device=z.device, dtype=Aij.dtype).index_add_(0, row, Aij)
        S = torch.zeros(shape, device=z.device, dtype=Sij.dtype).index_add_(0, row, Sij)

        # 标量调制
        norm = self.scalar_mlp(tensor_norm(I + A + S))
        norm = norm.view(-1, self.hidden_dim, 3)

        I = self.linear_I(I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * norm[..., 0, None, None]
        A = self.linear_A(A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * norm[..., 1, None, None]
        S = self.linear_S(S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * norm[..., 2, None, None]

        return I + A + S


class TensorInteraction(nn.Module):
    """张量交互层"""

    def __init__(
            self,
            hidden_dim: int,
            num_rbf: int,
            cutoff: float,
            equivariance_group: str,
            act_class,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.equivariance_group = equivariance_group

        self.cutoff = CosineCutoff(cutoff)

        # 标量路径
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            act_class(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            act_class(),
            nn.Linear(2 * hidden_dim, 3 * hidden_dim),
        )

        # 张量线性层
        self.linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(6)
        ])

    def forward(
            self,
            X: Tensor,
            edge_index: Tensor,
            edge_dist: Tensor,
            edge_attr: Tensor,
    ) -> Tensor:
        row, col = edge_index

        # 截断和边特征
        C = self.cutoff(edge_dist)
        edge_feat = self.scalar_mlp(edge_attr) * C.unsqueeze(-1)
        edge_feat = edge_feat.view(-1, self.hidden_dim, 3)

        # 归一化
        X = X / (tensor_norm(X) + 1).unsqueeze(-1).unsqueeze(-1)

        # 分解
        I, A, S = decompose_tensor(X)

        # 变换
        I = self.linears[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        Y = I + A + S

        # 消息传递
        Im = tensor_message(edge_index, edge_feat[..., 0, None, None], I, X.shape[0])
        Am = tensor_message(edge_index, edge_feat[..., 1, None, None], A, X.shape[0])
        Sm = tensor_message(edge_index, edge_feat[..., 2, None, None], S, X.shape[0])
        msg = Im + Am + Sm

        # 更新
        if self.equivariance_group == "O(3)":
            prod = torch.matmul(msg, Y) + torch.matmul(Y, msg)
        else:  # SO(3)
            prod = 2 * torch.matmul(Y, msg)

        I, A, S = decompose_tensor(prod)

        # 归一化
        norm = (tensor_norm(I + A + S) + 1).unsqueeze(-1).unsqueeze(-1)
        I, A, S = I / norm, A / norm, S / norm

        # 最终变换
        I = self.linears[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        dX = I + A + S
        return X + dX + torch.matrix_power(dX, 2)


# ============================================================================
# 辅助函数
# ============================================================================

def decompose_tensor(X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """将张量分解为 I (迹), A (反对称), S (无迹对称)"""
    # X: (batch, hidden, 3, 3)
    trace = X.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)  # (batch, hidden, 1)
    I = trace.unsqueeze(-1) / 3 * torch.eye(3, device=X.device)  # 迹部分
    A = (X - X.transpose(-1, -2)) / 2  # 反对称
    S = (X + X.transpose(-1, -2)) / 2 - I  # 无迹对称
    return I, A, S


def tensor_norm(X: Tensor) -> Tensor:
    """张量的 Frobenius 范数"""
    return (X ** 2).sum((-2, -1))


def vector_to_skew(v: Tensor) -> Tensor:
    """向量转反对称矩阵"""
    # v: (num_edges, 3)
    batch = v.shape[0]
    device = v.device
    dtype = v.dtype

    skew = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
    skew[:, 0, 1] = -v[:, 2]
    skew[:, 0, 2] = v[:, 1]
    skew[:, 1, 0] = v[:, 2]
    skew[:, 1, 2] = -v[:, 0]
    skew[:, 2, 0] = -v[:, 1]
    skew[:, 2, 1] = v[:, 0]

    return skew.unsqueeze(1)  # (E, 1, 3, 3)


def vector_to_sym(v: Tensor) -> Tensor:
    """向量转对称矩阵 (外积)"""
    # v: (num_edges, 3)
    return torch.einsum('ei,ej->eij', v, v).unsqueeze(1)  # (E, 1, 3, 3)


def tensor_message(edge_index, factor, tensor, num_atoms):
    """张量消息传递"""
    row, col = edge_index
    msg = factor * tensor[col]
    out = torch.zeros(num_atoms, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
    return out.index_add_(0, row, msg)