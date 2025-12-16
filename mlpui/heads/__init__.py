"""
MLPUI - Output Heads
====================

输出头模块 - 新属性只需在这里添加！

可用的属性:
- energy: 总能量
- atomic_energy: 每原子能量
- charges: 原子部分电荷
- dipole: 偶极矩
- dipole_magnitude: 偶极矩大小
- quadrupole: 四极矩
- polarizability: 极化率
- born_charges: Born 有效电荷
- homo_lumo_gap: HOMO-LUMO 能隙
- electron_density: 电子密度
"""

from mlpui.heads.properties import (
    # 注册表
    HEAD_REGISTRY,
    register_head,
    create_head,
    create_heads,
    list_available_heads,

    # 能量
    EnergyHead,
    AtomicEnergyHead,

    # 电荷和多极矩
    ChargesHead,
    DipoleHead,
    DipoleMagnitudeHead,
    QuadrupoleHead,

    # 响应性质
    PolarizabilityHead,
    BornChargesHead,

    # 电子性质
    HOMOLUMOGapHead,
    ElectronDensityHead,
)

__all__ = [
    # 工厂函数
    'HEAD_REGISTRY',
    'register_head',
    'create_head',
    'create_heads',
    'list_available_heads',

    # 输出头类
    'EnergyHead',
    'AtomicEnergyHead',
    'ChargesHead',
    'DipoleHead',
    'DipoleMagnitudeHead',
    'QuadrupoleHead',
    'PolarizabilityHead',
    'BornChargesHead',
    'HOMOLUMOGapHead',
    'ElectronDensityHead',
]