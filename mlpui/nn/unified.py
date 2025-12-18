"""
mlpui/nn/unified.py

UnifiedMLP - 统一的 MLP 组装器
================================

将 InputModule + Model + OutputHeads 组合成完整的端到端模型
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.core import BaseInputModule, BaseModel, BaseOutputHead, ModelInput, ModelOutput
from mlpui.inputs import create_input_module, INPUT_REGISTRY
from mlpui.models import create_model, MODEL_REGISTRY
from mlpui.outputs import create_output_head, OUTPUT_REGISTRY


class UnifiedMLP(nn.Module):
    """
    统一的机器学习势能模型

    将三个模块组合成完整的端到端模型:
    - InputModule: 原始数据 → ModelInput (图构建)
    - Model: ModelInput → ModelOutput (表示学习)
    - OutputHeads: ModelOutput → 目标属性 (属性预测)

    Args:
        input_module: InputModule 实例或配置
        model: Model 实例或配置
        output_heads: OutputHead 实例列表或配置

    Example:
        >>> # 方式 1: 传入实例
        >>> mlp = UnifiedMLP(
        ...     input_module=RadiusGraphInput(cutoff=5.0),
        ...     model=SchNet(hidden_dim=128),
        ...     output_heads=[EnergyHead(hidden_dim=128)],
        ... )

        >>> # 方式 2: 从配置创建
        >>> mlp = UnifiedMLP.from_config({
        ...     'input': {'type': 'radius', 'cutoff': 5.0},
        ...     'model': {'type': 'schnet', 'hidden_dim': 128},
        ...     'outputs': [{'type': 'energy'}],
        ... })

        >>> # 前向传播
        >>> results = mlp(batch)
        >>> energy = results['energy']
    """

    def __init__(
            self,
            input_module: BaseInputModule,
            model: BaseModel,
            output_heads: List[BaseOutputHead],
    ):
        super().__init__()

        self.input_module = input_module
        self.model = model

        # 将输出头存储为 ModuleDict，方便按名称访问
        self.output_heads = nn.ModuleDict({
            head.name: head for head in output_heads
        })

        # 验证兼容性
        self._validate_compatibility()

    def _validate_compatibility(self):
        """验证模块间的兼容性"""
        # 检查是否有输出头需要向量特征
        for name, head in self.output_heads.items():
            if hasattr(head, 'required_fields'):
                if 'vector_features' in head.required_fields:
                    if not self.model.has_vector_features:
                        raise ValueError(
                            f"OutputHead '{name}' requires vector_features, "
                            f"but model '{self.model.model_name}' does not provide them. "
                            f"Use a model with has_vector_features=True (e.g., NewtonNet, PaiNN)."
                        )

    @property
    def hidden_dim(self) -> int:
        """模型的隐藏维度"""
        return self.model.hidden_dim

    @property
    def cutoff(self) -> float:
        """输入模块的截断半径"""
        return self.input_module.cutoff

    @property
    def output_names(self) -> List[str]:
        """所有输出头的名称"""
        return list(self.output_heads.keys())

    def forward(
            self,
            batch: Any,
            compute_forces: bool = True,
            create_graph: bool = True,
    ) -> Dict[str, Tensor]:
        """
        完整的前向传播

        Args:
            batch: 数据批次 (PyG Data/Batch, dict, 或自定义对象)
            compute_forces: 是否计算力 (需要 energy_and_forces 头)
            create_graph: 是否创建计算图 (训练时需要)

        Returns:
            Dict[str, Tensor]: 包含所有预测属性的字典
        """
        # Stage 1: Input Module
        model_input = self.input_module(batch)

        # Stage 2: Model
        model_output = self.model(model_input)

        # Stage 3: Output Heads
        results = {}

        for name, head in self.output_heads.items():
            if name == 'energy_and_forces':
                # 特殊处理: 能量和力
                out = head(
                    model_output,
                    compute_forces=compute_forces,
                    create_graph=create_graph,
                )
                results.update(out)
            elif name == 'stress' or name == 'virials':
                # 应力需要能量和 displacement
                if 'energy' in results:
                    # 需要从 input_module 获取 displacement
                    displacement = getattr(model_input, 'displacement', None)
                    if displacement is not None:
                        out = head(
                            model_output,
                            energy=results['energy'],
                            displacement=displacement,
                            create_graph=create_graph,
                        )
                        results[name] = out
            else:
                # 普通输出头
                results[name] = head(model_output)

        return results

    def forward_representation(self, batch: Any) -> ModelOutput:
        """
        只计算表示，不计算输出属性

        用于提取特征、迁移学习等场景

        Args:
            batch: 数据批次

        Returns:
            ModelOutput: 模型的中间表示
        """
        model_input = self.input_module(batch)
        return self.model(model_input)

    def predict_property(
            self,
            batch: Any,
            property_name: str,
            **kwargs,
    ) -> Tensor:
        """
        预测单个属性

        Args:
            batch: 数据批次
            property_name: 属性名称
            **kwargs: 传给输出头的额外参数

        Returns:
            Tensor: 预测的属性值
        """
        if property_name not in self.output_heads:
            raise ValueError(
                f"Unknown property '{property_name}'. "
                f"Available: {self.output_names}"
            )

        model_input = self.input_module(batch)
        model_output = self.model(model_input)

        head = self.output_heads[property_name]
        return head(model_output, **kwargs)

    def add_output_head(self, head: BaseOutputHead, replace: bool = False):
        """
        添加新的输出头

        Args:
            head: OutputHead 实例
            replace: 是否替换同名的现有头
        """
        if head.name in self.output_heads and not replace:
            raise ValueError(
                f"OutputHead '{head.name}' already exists. "
                f"Use replace=True to replace it."
            )

        self.output_heads[head.name] = head
        self._validate_compatibility()

    def remove_output_head(self, name: str):
        """移除输出头"""
        if name not in self.output_heads:
            raise ValueError(f"OutputHead '{name}' not found.")
        del self.output_heads[name]

    def freeze_representation(self):
        """冻结表示层 (input_module + model)，只训练输出头"""
        for param in self.input_module.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_representation(self):
        """解冻表示层"""
        for param in self.input_module.parameters():
            param.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_output_heads(self, names: Optional[List[str]] = None):
        """冻结指定的输出头"""
        names = names or self.output_names
        for name in names:
            if name in self.output_heads:
                for param in self.output_heads[name].parameters():
                    param.requires_grad = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'UnifiedMLP':
        """
        从配置字典创建 UnifiedMLP

        Args:
            config: 配置字典，包含:
                - input: InputModule 配置
                - model: Model 配置
                - outputs: OutputHead 配置列表

        Example:
            >>> config = {
            ...     'input': {
            ...         'type': 'radius',
            ...         'cutoff': 5.0,
            ...     },
            ...     'model': {
            ...         'type': 'schnet',
            ...         'hidden_dim': 128,
            ...         'num_layers': 6,
            ...     },
            ...     'outputs': [
            ...         {'type': 'energy'},
            ...         {'type': 'forces'},
            ...     ],
            ... }
            >>> mlp = UnifiedMLP.from_config(config)
        """
        # 解析 input
        input_config = config.get('input', {})
        input_type = input_config.pop('type', 'radius')
        input_module = create_input_module(input_type, **input_config)

        # 解析 model
        model_config = config.get('model', {})
        model_type = model_config.pop('type', 'schnet')
        model = create_model(model_type, **model_config)

        # 解析 outputs
        output_configs = config.get('outputs', [{'type': 'energy'}])
        output_heads = []

        for out_config in output_configs:
            out_config = out_config.copy()
            out_type = out_config.pop('type')

            # 自动填充 hidden_dim
            if 'hidden_dim' not in out_config:
                out_config['hidden_dim'] = model.hidden_dim

            head = create_output_head(out_type, **out_config)
            output_heads.append(head)

        return cls(
            input_module=input_module,
            model=model,
            output_heads=output_heads,
        )

    @classmethod
    def from_pretrained(
            cls,
            checkpoint_path: str,
            map_location: str = 'cpu',
            strict: bool = True,
    ) -> 'UnifiedMLP':
        """
        从预训练检查点加载模型

        Args:
            checkpoint_path: 检查点路径
            map_location: 设备映射
            strict: 是否严格匹配参数

        Returns:
            UnifiedMLP 实例
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # 从检查点恢复配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = cls.from_config(config)
        else:
            raise ValueError(
                "Checkpoint does not contain 'config'. "
                "Cannot reconstruct model architecture."
            )

        # 加载权重
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        return model

    def save_checkpoint(
            self,
            path: str,
            config: Optional[Dict] = None,
            **extra_info,
    ):
        """
        保存检查点

        Args:
            path: 保存路径
            config: 模型配置 (可选)
            **extra_info: 额外信息 (epoch, optimizer 等)
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            **extra_info,
        }

        if config is not None:
            checkpoint['config'] = config

        torch.save(checkpoint, path)

    def __repr__(self) -> str:
        lines = [
            f"UnifiedMLP(",
            f"  input_module={self.input_module.__class__.__name__}(cutoff={self.cutoff}),",
            f"  model={self.model.model_name}(hidden_dim={self.hidden_dim}),",
            f"  output_heads={self.output_names},",
            f")",
        ]
        return '\n'.join(lines)


class EnergyForcesModel(UnifiedMLP):
    """
    预配置的能量-力模型

    最常用的配置，快速创建一个预测能量和力的模型

    Args:
        model_type: 模型类型
        hidden_dim: 隐藏维度
        num_layers: 层数
        cutoff: 截断半径
        **kwargs: 其他模型参数

    Example:
        >>> model = EnergyForcesModel('schnet', hidden_dim=128)
        >>> results = model(batch)
        >>> energy, forces = results['energy'], results['forces']
    """

    def __init__(
            self,
            model_type: str = 'schnet',
            hidden_dim: int = 128,
            num_layers: int = 6,
            cutoff: float = 5.0,
            **kwargs,
    ):
        from mlpui.inputs import RadiusGraphInput
        from mlpui.outputs import EnergyAndForcesHead

        input_module = RadiusGraphInput(cutoff=cutoff)

        model = create_model(
            model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            **kwargs,
        )

        output_heads = [
            EnergyAndForcesHead(hidden_dim=hidden_dim),
        ]

        super().__init__(
            input_module=input_module,
            model=model,
            output_heads=output_heads,
        )


class MultiPropertyModel(UnifiedMLP):
    """
    多属性预测模型

    同时预测多个属性 (能量、力、偶极矩、电荷等)

    Args:
        model_type: 模型类型
        hidden_dim: 隐藏维度
        num_layers: 层数
        cutoff: 截断半径
        properties: 要预测的属性列表
        **kwargs: 其他参数

    Example:
        >>> model = MultiPropertyModel(
        ...     'newtonnet',
        ...     properties=['energy', 'forces', 'dipole', 'charges'],
        ... )
    """

    def __init__(
            self,
            model_type: str = 'schnet',
            hidden_dim: int = 128,
            num_layers: int = 6,
            cutoff: float = 5.0,
            properties: List[str] = None,
            **kwargs,
    ):
        from mlpui.inputs import RadiusGraphInput

        properties = properties or ['energy', 'forces']

        input_module = RadiusGraphInput(cutoff=cutoff)

        model = create_model(
            model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            **kwargs,
        )

        # 构建输出头
        output_heads = []

        # 特殊处理: energy + forces 合并
        if 'energy' in properties and 'forces' in properties:
            from mlpui.outputs import EnergyAndForcesHead
            output_heads.append(EnergyAndForcesHead(hidden_dim=hidden_dim))
            properties = [p for p in properties if p not in ['energy', 'forces']]
        elif 'energy' in properties:
            from mlpui.outputs import EnergyHead
            output_heads.append(EnergyHead(hidden_dim=hidden_dim))
            properties.remove('energy')

        # 其他属性
        for prop in properties:
            head = create_output_head(prop, hidden_dim=hidden_dim)
            output_heads.append(head)

        super().__init__(
            input_module=input_module,
            model=model,
            output_heads=output_heads,
        )