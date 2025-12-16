"""
MLPUI - Output Heads (Property Predictors)
==========================================

🎯 关键模块：新属性只需在这里添加一次，所有模型自动支持！

使用方式:
    >>> from mlpui.heads import create_head, EnergyHead, DipoleHead
    >>>
    >>> # 创建单个头
    >>> energy_head = EnergyHead(hidden_dim=128)
    >>>
    >>> # 批量创建
    >>> heads = create_heads(['energy', 'forces', 'dipole'], hidden_dim=128)

添加新属性:
    @register_head('my_property')
    class MyPropertyHead(BaseOutputHead):
        def __init__(self, hidden_dim, **kwargs):
            ...

        @property
        def name(self):
            return 'my_property'

        def forward(self, mlp_output):
            # 计算属性
            return result
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.base import BaseOutputHead, MLPOutput, scatter_sum, scatter_mean

# =============================================================================
# 注册表
# =============================================================================

HEAD_REGISTRY: Dict[str, type] = {}


def register_head(name: str):
    """装饰器：注册输出头"""

    def decorator(cls):
        HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def create_head(name: str, hidden_dim: int, **kwargs) -> BaseOutputHead:
    """创建单个输出头"""
    if name not in HEAD_REGISTRY:
        available = list(HEAD_REGISTRY.keys())
        raise ValueError(f"Unknown head '{name}'. Available: {available}")
    return HEAD_REGISTRY[name](hidden_dim=hidden_dim, **kwargs)


def create_heads(
        names: List[str],
        hidden_dim: int,
        configs: Optional[Dict[str, Dict]] = None
) -> Dict[str, BaseOutputHead]:
    """批量创建输出头"""
    configs = configs or {}
    heads = {}
    for name in names:
        if name in ['forces', 'stress']:
            continue  # 这些通过梯度计算
        cfg = configs.get(name, {})
        heads[name] = create_head(name, hidden_dim, **cfg)
    return heads


def list_available_heads() -> List[str]:
    """列出所有可用的输出头"""
    return list(HEAD_REGISTRY.keys())




@register_head('energy')
class EnergyHead(BaseOutputHead):


    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            activation: str = 'silu',
            **kwargs
    ):
        super().__init__()

        act = nn.SiLU() if activation == 'silu' else nn.ReLU()

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'energy'

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        atomic_energies = self.mlp(mlp_output.node_features).squeeze(-1)
        energy = scatter_sum(atomic_energies, mlp_output.batch, dim=0)
        return energy


@register_head('atomic_energy')
class AtomicEnergyHead(BaseOutputHead):


    def __init__(self, hidden_dim: int, num_layers: int = 2, **kwargs):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'atomic_energy'

    @property
    def is_per_atom(self) -> bool:
        return True

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        return self.mlp(mlp_output.node_features).squeeze(-1)




@register_head('charges')
class ChargesHead(BaseOutputHead):


    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            enforce_neutrality: bool = False,
            total_charge: float = 0.0,
            **kwargs
    ):
        super().__init__()

        self.enforce_neutrality = enforce_neutrality
        self.total_charge = total_charge

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'charges'

    @property
    def is_per_atom(self) -> bool:
        return True

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        charges = self.mlp(mlp_output.node_features).squeeze(-1)

        if self.enforce_neutrality:
            num_atoms = scatter_sum(
                torch.ones_like(charges), mlp_output.batch, dim=0
            )
            total = scatter_sum(charges, mlp_output.batch, dim=0)
            mean = total / num_atoms
            charges = charges - mean[mlp_output.batch]

            if self.total_charge != 0.0:
                charges = charges + self.total_charge / num_atoms[mlp_output.batch]

        return charges


@register_head('dipole')
class DipoleHead(BaseOutputHead):


    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            use_center_of_mass: bool = True,
            **kwargs
    ):
        super().__init__()

        self.use_com = use_center_of_mass

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.charge_mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'dipole'

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        charges = self.charge_mlp(mlp_output.node_features)  # (N, 1)
        pos = mlp_output.pos  # (N, 3)
        batch = mlp_output.batch

        if self.use_com:
            num_atoms = scatter_sum(
                torch.ones(batch.shape[0], 1, device=pos.device), batch, dim=0
            )
            center = scatter_sum(pos, batch, dim=0) / num_atoms
            pos = pos - center[batch]

        dipole_contrib = charges * pos
        dipole = scatter_sum(dipole_contrib, batch, dim=0)

        return dipole


@register_head('dipole_magnitude')
class DipoleMagnitudeHead(BaseOutputHead):


    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__()
        self.dipole_head = DipoleHead(hidden_dim, **kwargs)

    @property
    def name(self) -> str:
        return 'dipole_magnitude'

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        dipole = self.dipole_head(mlp_output)
        return torch.norm(dipole, dim=-1)





@register_head('quadrupole')
class QuadrupoleHead(BaseOutputHead):


    def __init__(self, hidden_dim: int, num_layers: int = 2, **kwargs):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.charge_mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'quadrupole'

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        charges = self.charge_mlp(mlp_output.node_features)  # (N, 1)
        pos = mlp_output.pos
        batch = mlp_output.batch

        # 中心化
        num_atoms = scatter_sum(torch.ones_like(batch, dtype=pos.dtype), batch, dim=0)
        center = scatter_sum(pos, batch, dim=0) / num_atoms.unsqueeze(-1)
        r = pos - center[batch]

        r_sq = (r ** 2).sum(-1, keepdim=True)
        r_outer = r.unsqueeze(-1) * r.unsqueeze(-2)  # (N, 3, 3)

        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        identity_term = r_sq.unsqueeze(-1) * eye

        q_contrib = charges.unsqueeze(-1) * (3 * r_outer - identity_term)
        quadrupole = scatter_sum(q_contrib.view(-1, 9), batch, dim=0).view(-1, 3, 3)

        return quadrupole


# =============================================================================
# 极化率
# =============================================================================

@register_head('polarizability')
class PolarizabilityHead(BaseOutputHead):
    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 2,
            output_type: str = 'tensor',  # 'tensor', 'scalar', 'isotropic'
            **kwargs
    ):
        super().__init__()

        self.output_type = output_type
        out_dim = 6 if output_type == 'tensor' else 1  # 对称张量上三角

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'polarizability'

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        atomic = self.mlp(mlp_output.node_features)
        result = scatter_sum(atomic, mlp_output.batch, dim=0)

        if self.output_type == 'tensor':

            batch_size = result.shape[0]
            tensor = torch.zeros(batch_size, 3, 3, device=result.device, dtype=result.dtype)
            tensor[:, 0, 0] = result[:, 0]
            tensor[:, 0, 1] = tensor[:, 1, 0] = result[:, 1]
            tensor[:, 0, 2] = tensor[:, 2, 0] = result[:, 2]
            tensor[:, 1, 1] = result[:, 3]
            tensor[:, 1, 2] = tensor[:, 2, 1] = result[:, 4]
            tensor[:, 2, 2] = result[:, 5]
            return tensor

        return result.squeeze(-1)




@register_head('born_charges')
class BornChargesHead(BaseOutputHead):

    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__()
        self.dipole_head = DipoleHead(hidden_dim, use_center_of_mass=False, **kwargs)

    @property
    def name(self) -> str:
        return 'born_charges'

    @property
    def is_per_atom(self) -> bool:
        return True

    @property
    def requires_grad(self) -> bool:
        return True

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        pos = mlp_output.pos.requires_grad_(True)

        new_output = MLPOutput(
            node_features=mlp_output.node_features,
            z=mlp_output.z,
            pos=pos,
            batch=mlp_output.batch,
        )

        dipole = self.dipole_head(new_output)

        born = []
        for i in range(3):
            grad = torch.autograd.grad(
                dipole[:, i].sum(), pos,
                create_graph=True, retain_graph=True
            )[0]
            born.append(grad)

        return torch.stack(born, dim=-1)


@register_head('homo_lumo_gap')
class HOMOLUMOGapHead(BaseOutputHead):

    def __init__(self, hidden_dim: int, num_layers: int = 2, **kwargs):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'homo_lumo_gap'

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        atomic = self.mlp(mlp_output.node_features)
        return scatter_mean(atomic, mlp_output.batch, dim=0).squeeze(-1)


@register_head('electron_density')
class ElectronDensityHead(BaseOutputHead):

    def __init__(self, hidden_dim: int, num_layers: int = 2, **kwargs):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return 'electron_density'

    @property
    def is_per_atom(self) -> bool:
        return True

    def forward(self, mlp_output: MLPOutput) -> Tensor:
        return self.mlp(mlp_output.node_features).squeeze(-1)