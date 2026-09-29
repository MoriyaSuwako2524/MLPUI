"""Native UMA energy/gradient-force inference; no FAIRChem runtime dependency.

Energy head and normalization equations adapted from FAIRChem (MIT),
fairchem_core-2.17.0, commit be54a56. See LICENSE and docs/uma.md.
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn
from omegaconf import OmegaConf

# e3nn 0.4.4's bundled constants include the harmless built-in slice type.
with torch.serialization.safe_globals([slice]):
    from .uma import eSCNMDBackbone, eSCNMDMoeBackbone
from .inference import InferenceSettings


def plain(config):
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else copy.deepcopy(config)


class EnergyHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.energy_block = nn.Sequential(
            nn.Linear(backbone.sphere_channels, backbone.hidden_channels), nn.SiLU(),
            nn.Linear(backbone.hidden_channels, backbone.hidden_channels), nn.SiLU(),
            nn.Linear(backbone.hidden_channels, 1),
        )

    def forward(self, embedding, data):
        per_atom = self.energy_block(embedding[:, 0, :]).reshape(-1)
        return per_atom.new_zeros(len(data['natoms'])).index_add(0, data['batch'], per_atom)


class ExpertLinear(nn.Module):
    def __init__(self, layer, experts):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(experts, layer.out_features, layer.in_features))
        self.bias = nn.Parameter(torch.empty(layer.out_features))

    def forward(self, value, expert):
        return nn.functional.linear(value, self.weights[expert], self.bias)


class DatasetEnergyHead(nn.Module):
    def __init__(self, backbone, mapping):
        super().__init__()
        self.mapping = mapping
        targets = sorted(set(mapping.values()))
        self.indices = {name: targets.index(target) for name, target in mapping.items()}
        self.head = EnergyHead(backbone)
        for key, layer in enumerate(self.head.energy_block):
            if isinstance(layer, nn.Linear):
                self.head.energy_block[key] = ExpertLinear(layer, len(mapping))

    def forward(self, embedding, data):
        expert = self.indices[data['dataset'][0]]
        value = embedding[:, 0, :]
        for layer in self.head.energy_block:
            value = layer(value, expert) if isinstance(layer, ExpertLinear) else layer(value)
        return value.new_zeros(len(data['natoms'])).index_add(0, data['batch'], value.reshape(-1))


class NativeUMAModel(nn.Module):
    """Strictly restored backbone and energy head for conservative UMA models."""
    mlpui_family = 'uma'
    mlpui_native_uma = True

    def __init__(self, model_config):
        super().__init__()
        config = plain(model_config)
        backbone_config = dict(config['backbone'])
        name = backbone_config.pop('model', backbone_config.pop('_target_', ''))
        classes = {'eSCNMDMoeBackbone': eSCNMDMoeBackbone, 'eSCNMDBackbone': eSCNMDBackbone}
        if name.rsplit('.', 1)[-1] not in classes:
            raise ValueError(f'Unsupported UMA backbone: {name}')
        # Stress is not needed for molecular NVT. These flags do not alter weights.
        backbone_config.update(otf_graph=True, always_use_pbc=False,
                               regress_forces=True, direct_forces=False, regress_stress=False,
                               activation_checkpointing=False, use_quaternion_wigner=True,
                               radius_pbc_version=2, execution_mode='general')
        self.backbone = classes[name.rsplit('.', 1)[-1]](**backbone_config)
        heads = config.get('heads', {})
        if len(heads) != 1:
            raise ValueError('Native UMA currently requires exactly one MLP_EFS_Head')
        head_name, head_config = next(iter(heads.items()))
        head_config = dict(head_config)
        head_type = head_config.pop('module', head_config.pop('_target_', ''))
        if head_type.rsplit('.', 1)[-1] == 'DatasetSpecificMoEWrapper':
            if head_config.get('head_cls', '').rsplit('.', 1)[-1] != 'MLP_EFS_Head':
                raise ValueError('Unsupported dataset-specific UMA head')
            mapping = head_config['dataset_mapping']
            head = DatasetEnergyHead(self.backbone, mapping)
            head_config = head_config.get('head_kwargs', {})
        elif head_type.rsplit('.', 1)[-1] in ('MLP_EFS_Head', 'escnmd_efs_head'):
            head = EnergyHead(self.backbone)
        else:
            raise ValueError(f'Unsupported UMA head: {head_type}')
        if head_config.get('reduce', 'sum') != 'sum' or head_config.get('prefix'):
            raise ValueError('Only unprefixed sum-reduced UMA energy heads are supported')
        if set(head_config) - {'reduce', 'prefix', 'wrap_property'}:
            raise ValueError(f'Unsupported UMA head settings: {head_config}')
        self.output_heads = nn.ModuleDict({head_name: head})
        self.head_name = head_name
        self.tasks = []
        self._prepared = False

    def configure_tasks(self, tasks):
        self.tasks = [plain(task) for task in tasks]

    def task_config(self, dataset, prop):
        matches = [t for t in self.tasks if dataset in t.get('datasets', []) and t['property'] == prop]
        if len(matches) != 1:
            raise ValueError(f'Expected one {dataset}/{prop} task in UMA checkpoint, found {len(matches)}')
        return matches[0]

    def forward(self, data, dataset='omol'):
        if not self._prepared:
            settings = InferenceSettings(activation_checkpointing=False, merge_mole=False,
                compile=False, use_quaternion_wigner=True, internal_graph_gen_version=2)
            self.backbone.prepare_for_inference(data, settings)
            self._prepared = True
        with torch.enable_grad():
            emb = self.backbone(data)
            energy = self.output_heads[self.head_name](emb['node_embedding'], data)
            forces = -torch.autograd.grad(energy.sum(), data['pos'], create_graph=False)[0]
        result = {}
        for prop, value in [('energy', energy), ('forces', forces)]:
            task = self.task_config(dataset, prop)
            norm = task['normalizer']
            mean, scale = norm.get('mean', 0.0), norm.get('rmsd', 1.0)
            value = value * torch.as_tensor(scale, device=value.device, dtype=value.dtype)
            value = value + torch.as_tensor(mean, device=value.device, dtype=value.dtype)
            refs = task.get('element_references')
            if refs is not None:
                if prop != 'energy':
                    raise ValueError('Element references are supported only for energy')
                table_config = refs['element_references']
                if not isinstance(table_config, Mapping) or table_config.get('_target_') != 'torch.DoubleTensor':
                    raise ValueError('Expected literal DoubleTensor element references')
                table = torch.as_tensor(table_config['_args_'][0], device=value.device, dtype=torch.float64)
                offset = table.new_zeros(value.shape).index_add(0, data['batch'], table[data['atomic_numbers']])
                value = value + offset
            result[prop] = value.detach()
        return result


def load_native_uma(state, metadata, device=None, dtype=None):
    from mlpui.model_patcher import ModelPatcher

    model = NativeUMAModel(metadata['model_config'])
    state = {k: v for k, v in state.items() if k != 'n_averaged'}
    if state and all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.configure_tasks(metadata['tasks_config'])
    model.eval().requires_grad_(False)
    dtype = dtype or torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Native UMA supports float32 and float64 only')
    model.to(dtype=dtype)
    return ModelPatcher(model, load_device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')))
