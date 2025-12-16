

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from mlpui.base import BaseMLPWrapper, MLPOutput
from mlpui.mlps import create_mlp
from mlpui.heads import create_heads


class UnifiedMLP(nn.Module):


    def __init__(
            self,
            mlp: str | BaseMLPWrapper,
            properties: List[str] = ['energy', 'forces'],
            compute_forces: bool = True,
            compute_stress: bool = False,
            head_configs: Optional[Dict[str, Dict]] = None,
            **mlp_kwargs
    ):

        super().__init__()

        # 创建 MLP 表示层
        if isinstance(mlp, str):
            self.mlp = create_mlp(mlp, **mlp_kwargs)
        else:
            self.mlp = mlp

        # 创建输出头
        head_properties = [p for p in properties if p not in ['forces', 'stress']]
        if compute_forces and 'energy' not in head_properties:
            head_properties.insert(0, 'energy')

        self.heads = nn.ModuleDict(
            create_heads(head_properties, self.mlp.hidden_dim, head_configs)
        )

        self.properties = properties
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
            cell: Optional[Tensor] = None,
            **kwargs
    ) -> Dict[str, Tensor]:

        if self.compute_forces:
            pos = pos.requires_grad_(True)

        # 获取表示
        mlp_output = self.mlp(z, pos, batch, cell, **kwargs)

        # 计算属性
        outputs = {}

        for name, head in self.heads.items():
            if not head.requires_grad:
                outputs[name] = head(mlp_output)
                mlp_output.properties[name] = outputs[name]

        # 计算力
        if self.compute_forces and 'energy' in outputs:
            energy = outputs['energy']
            if energy.requires_grad or pos.requires_grad:
                forces = -torch.autograd.grad(
                    energy.sum(),
                    pos,
                    create_graph=self.training,
                    retain_graph=True
                )[0]
                outputs['forces'] = forces

        # 计算应力
        if self.compute_stress and cell is not None and 'energy' in outputs:
            # TODO: 实现应力计算
            pass

        # 计算需要梯度的属性
        for name, head in self.heads.items():
            if head.requires_grad:
                outputs[name] = head(mlp_output)

        return outputs

    @property
    def output_properties(self) -> List[str]:
        """可输出的属性列表"""
        props = list(self.heads.keys())
        if self.compute_forces:
            props.append('forces')
        if self.compute_stress:
            props.append('stress')
        return props


def create_unified_model(
        mlp: str,
        properties: List[str] = ['energy', 'forces'],
        **kwargs
) -> UnifiedMLP:
    return UnifiedMLP(mlp=mlp, properties=properties, **kwargs)