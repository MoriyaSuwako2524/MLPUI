"""QEq charge-electrostatics backend (E_charge of E = E_gas + E_charge + E_residual).

Reuses the TensorNet-QMMM dual-region sample/batch adapter (z, pos, mm_pos,
mm_charges, charge; ragged MM offsets). Labels must be environment parts,
e.g. energy = E_QMMM - E_gas, forces = F_QMMM - F_gas, while mm_potential /
mm_potential_gradient are the full QM/MM labels (E_gas has no MM dependence).
"""
import copy
import math

import torch

from .tensornet_qmmm import TensorNetQMMMBackend


class QEqBackend(TensorNetQMMMBackend):
    name = 'qeq_charge'
    aliases = ('qeq',)
    label = 'QEq charge electrostatics (E_charge)'
    web_supported = False
    supports_evaluation_artifacts = False
    training_targets = ('energy', 'forces', 'mm_forces', 'mm_potential', 'mm_potential_gradient',
                        'gas_mm_potential', 'gas_mm_potential_gradient', 'charges')
    supports_batching = True

    def metadata(self):
        return {**super().metadata(), 'architecture': 'qeq_tensornet_v1',
                'config_hint': 'Variational charge equilibration: TensorNet chi/J, Gaussian charges, MM external potential. '
                               'Energy = E_QEq(q*;phi) - E_QEq(q0;0); no dq/dR terms needed.'}

    def default_config(self):
        return dict(schema_version=1, architecture='qeq_tensornet_v1', precision=32, embedding_dimension=64,
                    num_layers=2, num_rbf=32, max_num_neighbors=64, cutoff_upper=5.0,
                    sigma_bounds=[0.3, 1.2], hardness_bounds=[2.0, 40.0],
                    units={'length': 'angstrom', 'energy': 'eV', 'charge': 'e'},
                    baseline={'mode': 'residual_only'})

    def _config(self, supplied):
        if supplied.get('schema_version', 1) != 1 or supplied.get('architecture', 'qeq_tensornet_v1') != 'qeq_tensornet_v1':
            raise ValueError('Incompatible QEq architecture')
        config = self.default_config()
        if set(supplied) - set(config):
            raise ValueError(f'Unknown QEq config fields: {set(supplied) - set(config)}')
        config.update(copy.deepcopy(supplied))
        if config['precision'] not in (32, 64):
            raise ValueError('QEq precision must be 32 or 64')
        for key in ('embedding_dimension', 'num_layers', 'num_rbf', 'max_num_neighbors'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not isinstance(config['cutoff_upper'], (int, float)) or not math.isfinite(config['cutoff_upper']) or config['cutoff_upper'] <= 0:
            raise ValueError('cutoff_upper must be finite and positive')
        for key in ('sigma_bounds', 'hardness_bounds'):
            low, high = config[key]
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
                raise ValueError(f'{key} must be [low, high] with 0 < low < high')
            config[key] = [float(low), float(high)]
        if config['units'] != {'length': 'angstrom', 'energy': 'eV', 'charge': 'e'}:
            raise ValueError('QEq requires angstrom/eV/e units after dataset conversion')
        if config['baseline'] != {'mode': 'residual_only'}:
            raise ValueError('QEq trains on environment labels supplied by the dataset (residual_only)')
        return config

    def configure_targets(self, config, targets):
        if set(targets) - set(self.training_targets):
            raise ValueError(f'Unsupported QEq target: {set(targets) - set(self.training_targets)}')
        return self._config(config)

    def add_charge_head(self, config):
        return config   # charges are intrinsic outputs of the equilibration

    def matches(self, state):
        if isinstance(state, torch.nn.Module):
            return type(state).__module__ == 'mlpui.models.qeq.model'
        return 'qeq_schema' in state

    def build_model(self, config):
        from mlpui.models.qeq.model import QEqCharge
        return QEqCharge(self._config(config))

    def load_model(self, state, config):
        if not config:
            raise ValueError('QEq loading requires model_config')
        if 'qeq_schema' not in state:
            raise ValueError('Not a QEq checkpoint')
        model = self.build_model(config)
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, model, inputs, properties):
        if set(properties) - set(self.training_targets):
            raise ValueError('Unsupported QEq property')
        return model(**inputs, properties=properties)
