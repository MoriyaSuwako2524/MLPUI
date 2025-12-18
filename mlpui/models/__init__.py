"""
mlpui/models/__init__.py

MLP 模型实现
============
"""

from .schnet import SchNet
from .tensornet import TensorNet
from .newtonnet import NewtonNet

from .layers import (
    AtomEmbedding, NodeEmbedding,
    GaussianRBF, BesselRBF, ExpNormRBF, get_rbf,
    CosineCutoff, PolynomialCutoff,
    get_activation,
)

# 模型注册表
MODEL_REGISTRY = {
    'schnet': SchNet,
    'tensornet': TensorNet,
    'newtonnet': NewtonNet,
}


def create_model(name: str, **kwargs):
    """
    工厂函数：根据名称创建模型

    Args:
        name: 模型名称
        **kwargs: 模型参数

    Returns:
        BaseModel 实例

    Example:
        >>> model = create_model('schnet', hidden_dim=128, num_layers=6)
        >>> model = create_model('tensornet', hidden_dim=256)
    """
    if name.lower() not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    return MODEL_REGISTRY[name.lower()](**kwargs)


def list_models():
    """列出所有可用的模型"""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    # 模型
    'schnet.py',
    'TensorNet',
    'NewtonNet',

    # 工厂
    'create_model',
    'list_models',
    'MODEL_REGISTRY',

    # 层
    'AtomEmbedding', 'NodeEmbedding',
    'GaussianRBF', 'BesselRBF', 'ExpNormRBF', 'get_rbf',
    'CosineCutoff', 'PolynomialCutoff',
    'get_activation',
]
