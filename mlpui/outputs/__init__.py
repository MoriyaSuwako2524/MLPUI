"""
mlpui/outputs/__init__.py

Output Heads
============

将 ModelOutput 转换为目标属性的输出头
"""

from .energy import (
    EnergyHead,
    EnergyAndForcesHead,
    DirectForceHead,
)
from mlpui.outputs.charges import ChargesHead
from .dipole import DipoleHead, DipoleFromChargesHead


# 注册表
OUTPUT_REGISTRY = {
    'energy': EnergyHead,
    'energy_and_forces': EnergyAndForcesHead,
    'forces': DirectForceHead,
    'charges': ChargesHead,
    'dipole': DipoleHead,
}


def create_output_head(name: str, hidden_dim: int, **kwargs):
    """
    工厂函数：根据名称创建 OutputHead

    Args:
        name: 头名称
        hidden_dim: 输入特征维度
        **kwargs: 其他参数

    Returns:
        BaseOutputHead 实例

    Example:
        >>> head = create_output_head('energy', hidden_dim=128)
        >>> head = create_output_head('dipole', hidden_dim=128, method='charge_based')
    """
    if name not in OUTPUT_REGISTRY:
        available = list(OUTPUT_REGISTRY.keys())
        raise ValueError(f"Unknown output head '{name}'. Available: {available}")

    return OUTPUT_REGISTRY[name](hidden_dim=hidden_dim, **kwargs)


def list_output_heads():
    """列出所有可用的 OutputHead"""
    return list(OUTPUT_REGISTRY.keys())


__all__ = [
    # 类
    'EnergyHead',
    'EnergyAndForcesHead',
    'DirectForceHead',
    'ChargesHead',
    'DipoleHead',
    'DipoleFromChargesHead',

    # 工厂函数
    'create_output_head',
    'list_output_heads',
    'OUTPUT_REGISTRY',
]
