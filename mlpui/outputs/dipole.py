"""
mlpui/outputs/dipole.py

偶极矩输出头
============

预测分子偶极矩
"""

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core import BaseOutputHead, ModelOutput, scatter_sum
from .charges import ChargesHead


class DipoleHead(BaseOutputHead):
    """
    偶极矩预测头

    两种计算方式:
    1. 基于电荷: μ = Σ q_i * (r_i - r_center)
    2. 直接预测: 使用向量特征直接预测

    Args:
        hidden_dim: 输入特征维度
        num_layers: MLP 层数
        activation: 激活函数
        method: 计算方式 ('charge_based' 或 'direct')
        use_center_of_mass: 是否使用质心作为参考点 (否则用几何中心)
        charge_conservation: 是否强制电荷守恒

    Example:
        >>> head = DipoleHead(hidden_dim=128, method='charge_based')
        >>> dipole = head(model_output)  # (num_graphs, 3)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            activation: str = 'silu',
            method: str = 'charge_based',
            use_center_of_mass: bool = False,
            charge_conservation: bool = True,
    ):
        super().__init__()

        self.method = method
        self.use_center_of_mass = use_center_of_mass

        if method == 'charge_based':
            # 预测电荷，然后计算偶极矩
            self.charge_head = ChargesHead(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
                charge_conservation=charge_conservation,
            )
        elif method == 'direct':
            # 直接预测原子偶极贡献
            act_fn = _get_activation(activation)
            layers = []
            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    act_fn,
                ])
            layers.append(nn.Linear(hidden_dim, 3))
            self.mlp = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'charge_based' or 'direct'")

    @property
    def name(self) -> str:
        return 'dipole'

    @property
    def is_per_atom(self) -> bool:
        return False

    @property
    def output_dim(self) -> int:
        return 3

    @property
    def required_fields(self) -> List[str]:
        if self.method == 'charge_based':
            return ['node_features', 'pos', 'batch']
        else:
            return ['node_features', 'batch']

    def forward(
            self,
            outputs: ModelOutput,
            total_charge: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算偶极矩

        Args:
            outputs: ModelOutput
            total_charge: 总电荷 (用于电荷守恒)

        Returns:
            dipole: (num_graphs, 3)
        """
        self.validate_input(outputs)

        batch = outputs.batch
        num_graphs = outputs.num_graphs

        if self.method == 'charge_based':
            return self._charge_based_dipole(outputs, total_charge)
        else:
            return self._direct_dipole(outputs)

    def _charge_based_dipole(
            self,
            outputs: ModelOutput,
            total_charge: Optional[Tensor],
    ) -> Tensor:
        """基于电荷计算偶极矩: μ = Σ q_i * (r_i - r_center)"""

        pos = outputs.pos
        batch = outputs.batch
        num_graphs = outputs.num_graphs
        z = outputs.z

        # 预测电荷
        charges = self.charge_head(outputs, total_charge)  # (num_atoms,)

        # 计算参考点
        if self.use_center_of_mass and z is not None:
            # 质心
            masses = _get_atomic_masses(z)
            center = _compute_center_of_mass(pos, masses, batch, num_graphs)
        else:
            # 几何中心
            center = _compute_center(pos, batch, num_graphs)

        # 相对位置
        rel_pos = pos - center[batch]  # (num_atoms, 3)

        # 偶极矩: μ = Σ q_i * r_i
        atomic_dipole = charges.unsqueeze(-1) * rel_pos  # (num_atoms, 3)
        dipole = scatter_sum(atomic_dipole, batch, dim=0, dim_size=num_graphs)

        return dipole

    def _direct_dipole(self, outputs: ModelOutput) -> Tensor:
        """直接预测偶极矩"""

        node_features = outputs.node_features
        batch = outputs.batch
        num_graphs = outputs.num_graphs

        # 预测原子偶极贡献
        atomic_dipole = self.mlp(node_features)  # (num_atoms, 3)

        # 聚合
        dipole = scatter_sum(atomic_dipole, batch, dim=0, dim_size=num_graphs)

        return dipole


class DipoleFromChargesHead(BaseOutputHead):
    """
    从外部电荷计算偶极矩

    当电荷由其他头预测时，可以用这个头计算偶极矩。

    Args:
        use_center_of_mass: 是否使用质心
    """

    def __init__(self, use_center_of_mass: bool = False):
        super().__init__()
        self.use_center_of_mass = use_center_of_mass

    @property
    def name(self) -> str:
        return 'dipole'

    @property
    def required_fields(self) -> List[str]:
        return ['pos', 'batch']

    def forward(
            self,
            outputs: ModelOutput,
            charges: Tensor,  # 外部传入的电荷
    ) -> Tensor:
        """
        从电荷计算偶极矩

        Args:
            outputs: ModelOutput
            charges: (num_atoms,) 原子电荷

        Returns:
            dipole: (num_graphs, 3)
        """
        pos = outputs.pos
        batch = outputs.batch
        num_graphs = outputs.num_graphs
        z = outputs.z

        if self.use_center_of_mass and z is not None:
            masses = _get_atomic_masses(z)
            center = _compute_center_of_mass(pos, masses, batch, num_graphs)
        else:
            center = _compute_center(pos, batch, num_graphs)

        rel_pos = pos - center[batch]
        atomic_dipole = charges.unsqueeze(-1) * rel_pos
        dipole = scatter_sum(atomic_dipole, batch, dim=0, dim_size=num_graphs)

        return dipole


# ============================================================================
# 辅助函数
# ============================================================================

def _compute_center(pos: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    """计算几何中心"""
    center = scatter_sum(pos, batch, dim=0, dim_size=num_graphs)
    counts = scatter_sum(torch.ones(pos.shape[0], device=pos.device), batch, dim=0, dim_size=num_graphs)
    return center / counts.unsqueeze(-1).clamp(min=1)


def _compute_center_of_mass(pos: Tensor, masses: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    """计算质心"""
    weighted_pos = pos * masses.unsqueeze(-1)
    total_mass = scatter_sum(masses, batch, dim=0, dim_size=num_graphs)
    com = scatter_sum(weighted_pos, batch, dim=0, dim_size=num_graphs)
    return com / total_mass.unsqueeze(-1).clamp(min=1e-8)


# 原子质量表 (简化版)
ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    15: 30.974, 16: 32.065, 17: 35.453, 35: 79.904, 53: 126.90,
}


def _get_atomic_masses(z: Tensor) -> Tensor:
    """获取原子质量"""
    masses = torch.ones_like(z, dtype=torch.float)
    for atomic_num, mass in ATOMIC_MASSES.items():
        masses[z == atomic_num] = mass
    return masses


def _get_activation(name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'gelu': nn.GELU(),
    }
    return activations.get(name.lower(), nn.SiLU())