"""
mlpui/outputs/energy.py

能量输出头
==========

预测分子/体系的总能量，可选计算力（通过能量梯度）
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core.base_output import BaseOutputHead, ModelOutput, scatter_sum


class EnergyHead(BaseOutputHead):
    """
    能量预测头

    将节点特征转换为原子能量，然后聚合为分子总能量。

    Args:
        hidden_dim: 输入特征维度 (来自 Model)
        num_layers: MLP 层数
        activation: 激活函数
        reduce: 聚合方式 ('sum' 或 'mean')

    Example:
        >>> head = EnergyHead(hidden_dim=128)
        >>> energy = head(model_output)  # (num_graphs,)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            activation: str = 'silu',
            reduce: str = 'sum',
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self.reduce = reduce

        # 激活函数
        act_fn = _get_activation(activation)

        # MLP: hidden_dim → hidden_dim → 1
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'energy'

    @property
    def is_per_atom(self) -> bool:
        return False  # 分子级属性

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def required_fields(self) -> List[str]:
        return ['node_features', 'batch']

    def forward(self, outputs: ModelOutput) -> Tensor:
        """
        计算能量

        Args:
            outputs: ModelOutput，包含 node_features 和 batch

        Returns:
            energy: (num_graphs,) 每个分子的能量
        """
        self.validate_input(outputs)

        # node_features: (num_atoms, hidden_dim)
        node_features = outputs.node_features
        batch = outputs.batch

        # 原子能量: (num_atoms, 1)
        atomic_energy = self.mlp(node_features)

        # 聚合为分子能量: (num_graphs,)
        if self.reduce == 'sum':
            energy = scatter_sum(atomic_energy.squeeze(-1), batch, dim=0, dim_size=outputs.num_graphs)
        else:  # mean
            energy = scatter_mean(atomic_energy.squeeze(-1), batch, dim=0, dim_size=outputs.num_graphs)

        return energy


class EnergyAndForcesHead(BaseOutputHead):
    """
    能量 + 力预测头

    能量通过 MLP 预测，力通过能量对位置的梯度计算。

    Args:
        hidden_dim: 输入特征维度
        num_layers: MLP 层数
        activation: 激活函数
        reduce: 能量聚合方式

    Note:
        使用此头时，需要确保 pos.requires_grad = True

    Example:
        >>> head = EnergyAndForcesHead(hidden_dim=128)
        >>> results = head(model_output)
        >>> energy = results['energy']   # (num_graphs,)
        >>> forces = results['forces']   # (num_atoms, 3)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            activation: str = 'silu',
            reduce: str = 'sum',
    ):
        super().__init__()

        self.energy_head = EnergyHead(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            reduce=reduce,
        )

    @property
    def name(self) -> str:
        return 'energy_and_forces'

    @property
    def required_fields(self) -> List[str]:
        return ['node_features', 'batch', 'pos']

    def forward(
            self,
            outputs: ModelOutput,
            compute_forces: bool = True,
            create_graph: bool = True,  # 训练时需要 True
    ) -> dict:
        """
        计算能量和力

        Args:
            outputs: ModelOutput
            compute_forces: 是否计算力
            create_graph: 是否创建计算图（训练时需要）

        Returns:
            dict: {'energy': Tensor, 'forces': Tensor}
        """
        self.validate_input(outputs)

        pos = outputs.pos

        # 确保位置需要梯度
        if compute_forces and not pos.requires_grad:
            pos.requires_grad_(True)

        # 计算能量
        energy = self.energy_head(outputs)

        results = {'energy': energy}

        # 计算力 (F = -∂E/∂pos)
        if compute_forces:
            # 对所有分子的能量求和，然后求梯度
            grad_outputs = torch.ones_like(energy)
            forces = -torch.autograd.grad(
                outputs=energy,
                inputs=pos,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            results['forces'] = forces

        return results


class DirectForceHead(BaseOutputHead):
    """
    直接力预测头 (使用等变向量特征)

    对于有向量特征的模型（NewtonNet, PaiNN），可以直接从向量特征预测力，
    不需要通过能量梯度。

    Args:
        hidden_dim: 输入特征维度
        num_layers: MLP 层数
        activation: 激活函数

    Note:
        需要模型输出 vector_features
    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            activation: str = 'silu',
    ):
        super().__init__()

        act_fn = _get_activation(activation)

        # MLP 处理标量特征，输出权重
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
            ])
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.scalar_mlp = nn.Sequential(*layers)

        # 最终投影到力
        self.force_proj = nn.Linear(hidden_dim, 1, bias=False)

    @property
    def name(self) -> str:
        return 'forces'

    @property
    def is_per_atom(self) -> bool:
        return True

    @property
    def output_dim(self) -> int:
        return 3

    @property
    def required_fields(self) -> List[str]:
        return ['node_features', 'vector_features']

    def forward(self, outputs: ModelOutput) -> Tensor:
        """
        计算力

        Args:
            outputs: ModelOutput，需要包含 vector_features

        Returns:
            forces: (num_atoms, 3)
        """
        self.validate_input(outputs)

        if not outputs.has_vector_features:
            raise ValueError(
                "DirectForceHead requires vector_features, but model does not provide them. "
                "Use EnergyAndForcesHead instead, or use a model with vector outputs."
            )

        node_features = outputs.node_features  # (num_atoms, hidden_dim)
        vector_features = outputs.vector_features  # (num_atoms, 3, hidden_dim)

        # 标量权重
        weights = self.scalar_mlp(node_features)  # (num_atoms, hidden_dim)

        # 加权向量特征
        # (num_atoms, 3, hidden_dim) * (num_atoms, 1, hidden_dim)
        weighted_vectors = vector_features * weights.unsqueeze(1)

        # 投影到力: (num_atoms, 3, hidden_dim) → (num_atoms, 3)
        forces = self.force_proj(weighted_vectors).squeeze(-1)

        return forces


def _get_activation(name: str) -> nn.Module:
    """获取激活函数"""
    activations = {
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]


def scatter_mean(src, index, dim=0, dim_size=None):
    """按索引求平均"""
    if dim_size is None:
        dim_size = int(index.max()) + 1

    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp(min=1)

    summed = scatter_sum(src, index, dim, dim_size)
    return summed / count