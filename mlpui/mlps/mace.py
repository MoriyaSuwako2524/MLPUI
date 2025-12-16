"""
MLPUI - MLP Wrappers Base
=========================

Base utilities for model wrappers.
"""

from typing import Dict, Type

from mlpui.base import BaseMLPWrapper

# 模型注册表
MLP_REGISTRY: Dict[str, Type[BaseMLPWrapper]] = {}


def register_mlp(name: str):
    """
    装饰器：注册新的 MLP 模型

    Example:
        @register_mlp('my_model')
        class MyModelWrapper(BaseMLPWrapper):
            ...
    """

    def decorator(cls: Type[BaseMLPWrapper]) -> Type[BaseMLPWrapper]:
        MLP_REGISTRY[name] = cls
        # 添加别名
        if hasattr(cls, 'aliases'):
            for alias in cls.aliases:
                MLP_REGISTRY[alias] = cls
        return cls

    return decorator


def get_mlp_class(name: str) -> Type[BaseMLPWrapper]:
    """根据名称获取 MLP 类"""
    if name not in MLP_REGISTRY:
        available = list(MLP_REGISTRY.keys())
        raise ValueError(f"Unknown MLP model '{name}'. Available: {available}")
    return MLP_REGISTRY[name]


def create_mlp(name: str, **kwargs) -> BaseMLPWrapper:
    """
    工厂函数：创建 MLP 模型

    Args:
        name: 模型名称 ('newtonnet', 'tensornet', 'mace' 等)
        **kwargs: 模型特定参数

    Returns:
        初始化的模型 wrapper

    Example:
        >>> mlp = create_mlp('tensornet', hidden_channels=128, num_layers=3)
    """
    cls = get_mlp_class(name)
    return cls(**kwargs)


def list_available_mlps() -> list:
    """列出所有可用的 MLP 模型"""
    return list(MLP_REGISTRY.keys())