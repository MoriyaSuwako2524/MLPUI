

__version__ = "0.1.0"

# 基础类
from mlpui.base import (
    BaseMLPWrapper,
    BaseOutputHead,
    MLPOutput,
    scatter_sum,
    scatter_mean,
)

# MLP 模型
from mlpui.mlps import (
    create_mlp,
    list_available_mlps,
    MLP_REGISTRY,
    register_mlp,
)

# 输出头
from mlpui.heads import (
    create_head,
    create_heads,
    list_available_heads,
    HEAD_REGISTRY,
    register_head,
    EnergyHead,
    ChargesHead,
    DipoleHead,
    PolarizabilityHead,
)

# 统一模型
from mlpui.model import UnifiedMLP, create_unified_model

# PyTorch Lightning
from mlpui.lightning import MLPLitModule, MLPDataModule


# 便捷函数
def create_model(
        mlp: str,
        properties: list = ['energy', 'forces'],
        **kwargs
) -> UnifiedMLP:
    """
    创建统一 MLP 模型

    Args:
        mlp: 模型类型 ('tensornet', 'newtonnet', 'mace' 等)
        properties: 属性列表 ('energy', 'forces', 'dipole' 等)
        **kwargs: 模型参数

    Returns:
        UnifiedMLP 模型实例

    Example:
        >>> model = create_model(
        ...     'tensornet',
        ...     properties=['energy', 'forces', 'dipole'],
        ...     hidden_channels=128,
        ...     num_layers=3,
        ... )
    """
    return create_unified_model(mlp, properties, **kwargs)


def create_trainer(
        mlp: str,
        properties: list = ['energy', 'forces'],
        loss_weights: dict = None,
        **kwargs
) -> MLPLitModule:
    """
    创建 PyTorch Lightning 训练模块

    Args:
        mlp: 模型类型
        properties: 属性列表
        loss_weights: 损失权重
        **kwargs: 其他参数

    Returns:
        MLPLitModule 实例
    """
    return MLPLitModule(
        mlp=mlp,
        properties=properties,
        loss_weights=loss_weights,
        **kwargs
    )


__all__ = [
    # 版本
    '__version__',

    # 基础类
    'BaseMLPWrapper',
    'BaseOutputHead',
    'MLPOutput',
    'scatter_sum',
    'scatter_mean',

    # MLP
    'create_mlp',
    'list_available_mlps',
    'MLP_REGISTRY',
    'register_mlp',

    # 输出头
    'create_head',
    'create_heads',
    'list_available_heads',
    'HEAD_REGISTRY',
    'register_head',
    'EnergyHead',
    'ChargesHead',
    'DipoleHead',
    'PolarizabilityHead',

    # 统一模型
    'UnifiedMLP',
    'create_unified_model',
    'create_model',

    # Lightning
    'MLPLitModule',
    'MLPDataModule',
    'create_trainer',
]