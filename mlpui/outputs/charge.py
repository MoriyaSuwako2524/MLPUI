"""
mlpui/outputs/charges.py

电荷输出头
==========

预测原子电荷，可选电荷守恒约束
"""

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core.base_output import BaseOutputHead, ModelOutput, scatter_sum, scatter_mean


class ChargesHead(BaseOutputHead):
    """
    原子电荷预测头

    Args:
        hidden_dim: 输入特征维度
        num_layers: MLP 层数
        activation: 激活函数
        total_charge: 总电荷约束 (None 表示不约束)
        charge_conservation: 是否强制电荷守恒 (每个分子电荷和为 total_charge)

    Example:
        >>> head = ChargesHead(hidden_dim=128, charge_conservation=True)
        >>> charges = head(model_output)  # (num_atoms,)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            activation: str = 'silu',
            total_charge: Optional[float] = 0.0,
            charge_conservation: bool = True,
    ):
        super().__init__()

        self.total_charge = total_charge
        self.charge_conservation = charge_conservation

        act_fn = _get_activation(activation)

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
        return 'charges'

    @property
    def is_per_atom(self) -> bool:
        return True

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def required_fields(self) -> List[str]:
        return ['node_features', 'batch']

    def forward(
            self,
            outputs: ModelOutput,
            total_charge: Optional[Tensor] = None,  # (num_graphs,) 可以外部传入
    ) -> Tensor:
        """
        计算原子电荷

        Args:
            outputs: ModelOutput
            total_charge: 每个分子的总电荷，如果需要强制守恒

        Returns:
            charges: (num_atoms,)
        """
        self.validate_input(outputs)

        node_features = outputs.node_features
        batch = outputs.batch
        num_graphs = outputs.num_graphs

        # 预测原始电荷
        charges = self.mlp(node_features).squeeze(-1)  # (num_atoms,)

        # 电荷守恒约束
        if self.charge_conservation:
            if total_charge is None:
                if self.total_charge is not None:
                    # 使用默认总电荷
                    total_charge = torch.full(
                        (num_graphs,), self.total_charge,
                        dtype=charges.dtype, device=charges.device
                    )
                else:
                    # 不强制约束
                    return charges

            # 计算当前总电荷
            current_total = scatter_sum(charges, batch, dim=0, dim_size=num_graphs)

            # 计算每个原子的修正量
            num_atoms_per_mol = scatter_sum(
                torch.ones_like(charges), batch, dim=0, dim_size=num_graphs
            )
            correction = (total_charge - current_total) / num_atoms_per_mol

            # 应用修正
            charges = charges + correction[batch]

        return charges


def _get_activation(name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
    }
    return activations.get(name.lower(), nn.SiLU())