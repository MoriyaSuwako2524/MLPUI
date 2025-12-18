from mlpui.inputs.radius_graph import RadiusGraphInput, radius_graph
from mlpui.inputs.knn_graph import KNNGraphInput, knn_graph


INPUT_REGISTRY = {
    'radius': RadiusGraphInput,
    'radius_graph': RadiusGraphInput,
    'knn': KNNGraphInput,
    'knn_graph': KNNGraphInput,
}


def create_input_module(name: str, **kwargs):
    """

    Args:
        name: 模块名称 ('radius', 'knn', 'periodic')
        **kwargs: 传递给构造函数的参数

    Returns:
        BaseInputModule 实例

    Example:
        >>> input_module = create_input_module('radius', cutoff=5.0)
        >>> input_module = create_input_module('periodic', cutoff=6.0, pbc=[True, True, False])
    """
    if name not in INPUT_REGISTRY:
        available = list(INPUT_REGISTRY.keys())
        raise ValueError(f"Unknown input module '{name}'. Available: {available}")

    return INPUT_REGISTRY[name](**kwargs)


def list_input_modules():
    """列出所有可用的 InputModule"""
    return list(set(INPUT_REGISTRY.values()))


__all__ = [
    # 类
    'RadiusGraphInput',
    'KNNGraphInput',


    # 工厂函数
    'create_input_module',
    'list_input_modules',
    'INPUT_REGISTRY',

    # 底层函数
    'radius_graph',
    'knn_graph',
]
