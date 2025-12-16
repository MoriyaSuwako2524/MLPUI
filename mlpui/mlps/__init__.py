

from mlpui.mlps.base import (
    MLP_REGISTRY,
    register_mlp,
    get_mlp_class,
    create_mlp,
    list_available_mlps,
)

# 导入具体实现以触发注册
from mlpui.mlps import newtonnet
from mlpui.mlps import tensornet
from mlpui.mlps import mace

# 导入具体类
from mlpui.mlps.newtonnet import NewtonNetWrapper, NewtonNetFromCheckpoint
from mlpui.mlps.tensornet import (
    TensorNetWrapper,
    EquivariantTransformerWrapper,
    GraphNetworkWrapper,
    TorchMDNetFromCheckpoint,
)
from mlpui.mlps.mace import MACEWrapper, MACEMPWrapper

__all__ = [
    # 工厂函数
    'create_mlp',
    'get_mlp_class',
    'list_available_mlps',
    'register_mlp',
    'MLP_REGISTRY',

    # NewtonNet
    'NewtonNetWrapper',
    'NewtonNetFromCheckpoint',

    # TorchMD-Net
    'TensorNetWrapper',
    'EquivariantTransformerWrapper',
    'GraphNetworkWrapper',
    'TorchMDNetFromCheckpoint',

    # MACE
    'MACEWrapper',
    'MACEMPWrapper',
]