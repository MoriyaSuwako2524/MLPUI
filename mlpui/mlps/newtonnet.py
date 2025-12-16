"""
MLPUI - NewtonNet Wrapper
=========================

Adapter for NewtonNet models.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.base import BaseMLPWrapper, MLPOutput
from mlpui.mlps.base import register_mlp


@register_mlp('newtonnet')
class NewtonNetWrapper(BaseMLPWrapper):

    aliases = ['newton', 'newtonnet-wrapper']

    def __init__(
            self,
            n_features: int = 128,
            n_interactions: int = 3,
            n_basis: int = 32,
            cutoff: float = 5.0,
            activation: str = 'silu',
            layer_norm: bool = False,
            pretrained: Optional[str] = None,
            **kwargs
    ):
        super().__init__()

        self._hidden_dim = n_features
        self.cutoff = cutoff

        try:
            from newtonnet.models.newtonnet import EmbeddingNet, InteractionNet
            from newtonnet.layers.activations import get_activation_by_string
        except ImportError:
            raise ImportError(
                "NewtonNet not installed. Install with: pip install newtonnet"
            )

        activation_fn = get_activation_by_string(activation)

        # 构建嵌入层
        self.embedding = EmbeddingNet(
            cutoff=cutoff,
            n_features=n_features,
            n_basis=n_basis,
        )

        # 构建交互层
        self.interactions = nn.ModuleList([
            InteractionNet(
                n_features=n_features,
                n_basis=n_basis,
                activation=activation_fn,
                layer_norm=layer_norm,
            ) for _ in range(n_interactions)
        ])

        # 加载预训练权重
        if pretrained:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 过滤只保留 embedding 和 interaction 层
        filtered = {}
        for key, value in state_dict.items():
            if key.startswith('embedding_layers.'):
                new_key = key.replace('embedding_layers.', 'embedding.')
                filtered[new_key] = value
            elif key.startswith('interaction_layers.'):
                new_key = key.replace('interaction_layers.', 'interactions.')
                filtered[new_key] = value

        self.load_state_dict(filtered, strict=False)
        print(f"Loaded pretrained NewtonNet from {path}")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "NewtonNet"

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """前向传播"""
        # 处理 cell
        if cell is None:
            num_graphs = int(batch.max()) + 1
            cell = torch.zeros(num_graphs, 3, 3, device=pos.device, dtype=pos.dtype)

        # 嵌入
        atom_node, force_node, dir_edge, dist_edge, edge_index, displacement = \
            self.embedding(z, pos, cell, batch)

        # 消息传递
        for interaction in self.interactions:
            atom_node, force_node = interaction(
                atom_node, force_node, dir_edge, dist_edge, edge_index
            )

        return MLPOutput(
            node_features=atom_node,
            vector_features=force_node,
            z=z,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            cell=cell,
            displacement=displacement,
        )


@register_mlp('newtonnet-checkpoint')
class NewtonNetFromCheckpoint(BaseMLPWrapper):
    """
    从 checkpoint 加载完整的 NewtonNet 模型

    用于已有训练好的模型，想要添加新的输出头时使用。

    Args:
        checkpoint_path: checkpoint 文件路径
        freeze: 是否冻结表示层权重
    """

    def __init__(
            self,
            checkpoint_path: str,
            freeze: bool = False,
            **kwargs
    ):
        super().__init__()

        try:
            from newtonnet.models.newtonnet import NewtonNet
        except ImportError:
            raise ImportError("NewtonNet not installed.")

        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 获取配置
        config = checkpoint.get('config', {})
        self._hidden_dim = config.get('n_features', 128)

        # 创建并加载模型
        model = NewtonNet(**{k: v for k, v in config.items()
                             if k not in ['output_properties']})

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # 提取表示层
        self.embedding = model.embedding_layers
        self.interactions = model.interaction_layers

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def has_vector_features(self) -> bool:
        return True

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> MLPOutput:
        """前向传播"""
        if cell is None:
            num_graphs = int(batch.max()) + 1
            cell = torch.zeros(num_graphs, 3, 3, device=pos.device, dtype=pos.dtype)

        atom_node, force_node, dir_edge, dist_edge, edge_index, displacement = \
            self.embedding(z, pos, cell, batch)

        for interaction in self.interactions:
            atom_node, force_node = interaction(
                atom_node, force_node, dir_edge, dist_edge, edge_index
            )

        return MLPOutput(
            node_features=atom_node,
            vector_features=force_node,
            z=z,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            cell=cell,
            displacement=displacement,
        )