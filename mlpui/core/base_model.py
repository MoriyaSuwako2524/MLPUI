"""
mlpui/core/base_model.py

Model 基类
==========

定义了 MLP 表示模型的标准接口
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn

from mlpui.core.types import ModelInput, ModelOutput


class BaseModel(nn.Module, ABC):
    """
    MLP 表示模型的抽象基类

    职责:
      - 接收 ModelInput (原子信息 + 图结构)
      - 输出 ModelOutput (节点特征)
      - 只做表示学习，不做属性预测

    所有具体模型 (TensorNet, NewtonNet, MACE, ...) 都继承此类

    子类必须实现:
      - forward(): 前向传播
      - hidden_dim: 输出特征维度

    子类可选实现:
      - has_vector_features: 是否输出向量特征
      - model_name: 模型名称

    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self, hidden_dim=128):
        ...         super().__init__()
        ...         self._hidden_dim = hidden_dim
        ...         # ... 构建网络层 ...
        ...
        ...     @property
        ...     def hidden_dim(self) -> int:
        ...         return self._hidden_dim
        ...
        ...     def forward(self, inputs: ModelInput) -> ModelOutput:
        ...         # ... 计算节点特征 ...
        ...         return ModelOutput.from_input(inputs, node_features)
    """

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """
        输出特征的维度

        Returns:
            int: node_features 的最后一个维度大小
        """
        pass

    @property
    def has_vector_features(self) -> bool:
        """
        是否输出等变向量特征

        等变模型 (NewtonNet, MACE, PaiNN) 返回 True
        不变模型 (SchNet, DimeNet) 返回 False

        Returns:
            bool: 默认 False
        """
        return False

    @property
    def model_name(self) -> str:
        """
        模型名称，用于日志和检查点

        Returns:
            str: 默认返回类名
        """
        return self.__class__.__name__

    @abstractmethod
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """
        前向传播

        Args:
            inputs: 标准化的模型输入 (ModelInput)
                - inputs.z: 原子序数
                - inputs.pos: 原子位置
                - inputs.batch: 批次索引
                - inputs.edge_index: 边索引
                - inputs.edge_dist: 边距离
                - inputs.edge_vec: 边向量
                - ...

        Returns:
            ModelOutput: 标准化的模型输出
                - node_features: (num_atoms, hidden_dim)
                - vector_features: (num_atoms, 3, hidden_dim) 可选
                - 以及透传的输入信息

        Note:
            推荐使用 ModelOutput.from_input() 创建输出，
            以确保必要的信息被正确透传
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.model_name}("
            f"hidden_dim={self.hidden_dim}, "
            f"has_vector_features={self.has_vector_features})"
        )

