"""
MLPUI - TorchMD-Net Wrapper
===========================

Adapters for TorchMD-Net models (TensorNet, ET, Graph Network).
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.base import BaseMLPWrapper, MLPOutput
from mlpui.mlps.base import register_mlp


@register_mlp('tensornet')
class TensorNetWrapper(BaseMLPWrapper):

    aliases = ['tensor-net', 'torchmd-tensornet']

    def __init__(
            self,
            hidden_channels: int = 128,
            num_layers: int = 2,
            num_rbf: int = 32,
            rbf_type: str = 'expnorm',
            trainable_rbf: bool = False,
            activation: str = 'silu',
            cutoff_lower: float = 0.0,
            cutoff_upper: float = 5.0,
            max_z: int = 100,
            max_num_neighbors: int = 64,
            equivariance_invariance_group: str = 'O(3)',
            pretrained: Optional[str] = None,
            **kwargs
    ):
        super().__init__()

        self._hidden_dim = hidden_channels

        try:
            from torchmdnet.models.tensornet import TensorNet
        except ImportError:
            raise ImportError(
                "TorchMD-Net not installed. Install with: pip install torchmd-net"
            )

        self.model = TensorNet(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=False,
            check_errors=True,
            box_vecs=None,
            dtype=torch.float32,
        )

        if pretrained:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 过滤表示模型的权重
        filtered = {}
        prefixes = ['model.representation_model.', 'representation_model.']
        for k, v in state_dict.items():
            for prefix in prefixes:
                if k.startswith(prefix):
                    new_key = 'model.' + k[len(prefix):]
                    filtered[new_key] = v
                    break

        if filtered:
            self.load_state_dict(filtered, strict=False)
            print(f"Loaded pretrained TensorNet from {path}")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "TensorNet"

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """前向传播"""
        scalar_features, vector_features, z_out, pos_out, batch_out = \
            self.model(z, pos, batch, box=cell)

        return MLPOutput(
            node_features=scalar_features,
            vector_features=vector_features,
            z=z_out,
            pos=pos_out,
            batch=batch_out,
            cell=cell,
        )


@register_mlp('equivariant-transformer')
class EquivariantTransformerWrapper(BaseMLPWrapper):
    """
    Equivariant Transformer (ET) 模型适配器

    基于注意力机制的等变模型。
    """

    aliases = ['et', 'torchmd-et']

    def __init__(
            self,
            hidden_channels: int = 128,
            num_layers: int = 6,
            num_rbf: int = 64,
            rbf_type: str = 'expnorm',
            trainable_rbf: bool = False,
            activation: str = 'silu',
            cutoff_lower: float = 0.0,
            cutoff_upper: float = 5.0,
            max_z: int = 100,
            max_num_neighbors: int = 64,
            num_heads: int = 8,
            distance_influence: str = 'both',
            pretrained: Optional[str] = None,
            **kwargs
    ):
        super().__init__()

        self._hidden_dim = hidden_channels

        try:
            from torchmdnet.models.torchmd_et import TorchMD_ET
        except ImportError:
            raise ImportError("TorchMD-Net not installed.")

        self.model = TorchMD_ET(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            attn_activation=activation,
            num_heads=num_heads,
            distance_influence=distance_influence,
            neighbor_embedding=True,
            check_errors=True,
            box_vecs=None,
            dtype=torch.float32,
        )

        if pretrained:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        filtered = {}
        for k, v in state_dict.items():
            if 'representation_model.' in k:
                new_key = k.split('representation_model.')[-1]
                filtered['model.' + new_key] = v

        if filtered:
            self.load_state_dict(filtered, strict=False)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "EquivariantTransformer"

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """前向传播"""
        scalar_features, vector_features, z_out, pos_out, batch_out = \
            self.model(z, pos, batch, box=cell)

        return MLPOutput(
            node_features=scalar_features,
            vector_features=vector_features,
            z=z_out,
            pos=pos_out,
            batch=batch_out,
            cell=cell,
        )


@register_mlp('graph-network')
class GraphNetworkWrapper(BaseMLPWrapper):
    """
    Graph Network 模型适配器

    不变的 SchNet 类模型。
    """

    aliases = ['gn', 'schnet-like']

    def __init__(
            self,
            hidden_channels: int = 128,
            num_layers: int = 6,
            num_rbf: int = 64,
            cutoff_upper: float = 5.0,
            max_z: int = 100,
            max_num_neighbors: int = 64,
            pretrained: Optional[str] = None,
            **kwargs
    ):
        super().__init__()

        self._hidden_dim = hidden_channels

        try:
            from torchmdnet.models.torchmd_gn import TorchMD_GN
        except ImportError:
            raise ImportError("TorchMD-Net not installed.")

        self.model = TorchMD_GN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff_lower=0.0,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            num_filters=hidden_channels,
            aggr='add',
            neighbor_embedding=True,
            check_errors=True,
            box_vecs=None,
            dtype=torch.float32,
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "GraphNetwork"

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """前向传播"""
        scalar_features, vector_features, z_out, pos_out, batch_out = \
            self.model(z, pos, batch, box=cell)

        return MLPOutput(
            node_features=scalar_features,
            vector_features=vector_features,
            z=z_out,
            pos=pos_out,
            batch=batch_out,
            cell=cell,
        )


@register_mlp('torchmd-checkpoint')
class TorchMDNetFromCheckpoint(BaseMLPWrapper):
    """
    从 TorchMD-Net checkpoint 加载模型
    """

    def __init__(
            self,
            checkpoint_path: str,
            freeze: bool = False,
            **kwargs
    ):
        super().__init__()

        try:
            from torchmdnet.models.model import load_model
        except ImportError:
            raise ImportError("TorchMD-Net not installed.")

        full_model = load_model(checkpoint_path)
        self.model = full_model.representation_model
        self._hidden_dim = self.model.hidden_channels

        model_class = type(self.model).__name__
        self._has_vector = model_class in ['TensorNet', 'TorchMD_ET']

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return self._has_vector

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """前向传播"""
        scalar_features, vector_features, z_out, pos_out, batch_out = \
            self.model(z, pos, batch, box=cell)

        return MLPOutput(
            node_features=scalar_features,
            vector_features=vector_features,
            z=z_out,
            pos=pos_out,
            batch=batch_out,
            cell=cell,
        )