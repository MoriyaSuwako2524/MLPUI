"""The user-facing tensornet-ext option restores the historical QM-only H+3 model.

It is NOT the upstream panxl TensorNet-ext architecture used by TensorNet-QMMM.
"""
import copy
import math
import re
import torch
from .tensornet_qmmm import TensorNetQMMMBackend


class TensorNetLegacyBackend(TensorNetQMMMBackend):
    name = 'tensornet-ext'
    aliases = ('tensornet_ext',)
    label = 'tensornet-ext (legacy QM-only + H+3)'

    def metadata(self):
        return {**super().metadata(), 'architecture': 'tensornet_qm_h3_v1',
                'config_hint': 'Historical schema1 QM-only TensorNet + H+3 electrostatic head. This comparison option is distinct from the upstream panxl TensorNet-ext radial MM messages used by TensorNet-QMMM. Same four supervision targets and baseline contract.'}

    def default_config(self):
        return dict(schema_version=1, architecture='tensornet_qm_h3_v1', precision=32, embedding_dimension=64, num_layers=3,
            num_rbf=32, max_num_neighbors=64, cutoff_upper=5., environment_cutoff=10.,
            environment_mode='fixed_reference_selection', potential_scale=1., field_scale=1.,
            units={'length': 'angstrom', 'energy': 'eV', 'charge': 'e'},
            baseline={'mode': 'residual_only'})

    def _config(self, supplied):
        config = self.default_config()
        if set(supplied) - set(config):
            raise ValueError(f'Unknown tensornet-ext config fields: {set(supplied) - set(config)}')
        config.update(copy.deepcopy(supplied))
        if config['architecture'] != 'tensornet_qm_h3_v1':
            raise ValueError('Incompatible tensornet-ext architecture')
        if type(config['schema_version']) is not int or config['schema_version'] != 1:
            raise ValueError('Unsupported tensornet-ext schema_version')
        if config['precision'] not in (32, 64):
            raise ValueError('QMMM precision must be 32 or 64')
        for key in ('embedding_dimension', 'num_layers', 'num_rbf', 'max_num_neighbors'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('cutoff_upper', 'potential_scale', 'field_scale'):
            if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if config['environment_cutoff'] != 10. or config['environment_mode'] != 'fixed_reference_selection':
            raise ValueError('First-release QMMM requires a fixed reference-selected 10 A environment')
        if config['units'] != {'length': 'angstrom', 'energy': 'eV', 'charge': 'e'}:
            raise ValueError('QMMM requires angstrom/eV/e units after dataset conversion')
        baseline = config['baseline']
        if not isinstance(baseline, dict):
            raise ValueError('baseline must be a configuration object')
        if baseline.get('mode') == 'residual_only':
            if set(baseline) != {'mode'}:
                raise ValueError('Residual-only baseline accepts only mode')
        elif baseline.get('mode') == 'offline':
            required = {'mode', 'energy_definition', 'qm_force_kind', 'uma_checkpoint_sha256',
                        'charge_checkpoint_sha256', 'source_manifest_sha256'}
            if set(baseline) != required or baseline['energy_definition'] != 'uma_gas_plus_gas_charge_coulomb':
                raise ValueError('Offline baseline requires exact energy definition and provenance metadata')
            if baseline['qm_force_kind'] not in ('full_dq_dR', 'fixed_charge_approximation'):
                raise ValueError('Offline qm_force_kind must explicitly declare dq/dR treatment')
            for key in ('uma_checkpoint_sha256', 'charge_checkpoint_sha256', 'source_manifest_sha256'):
                if not isinstance(baseline[key], str) or not re.fullmatch('[0-9a-f]{64}', baseline[key]):
                    raise ValueError(f'{key} must be a lowercase SHA256 digest')
        else:
            raise ValueError('Supported baseline modes are residual_only and offline; online UMA is not connected')
        return config

    def matches(self, state):
        if isinstance(state, torch.nn.Module):
            return type(state).__module__ == 'mlpui.models.tensornet_legacy.model'
        schema = state.get('qmmm_schema')
        return isinstance(schema, torch.Tensor) and schema.numel() == 1 and schema.item() == 1

    def build_model(self, config):
        from mlpui.models.tensornet_legacy.model import TensorNetLegacy
        return TensorNetLegacy(self._config(config))

    def load_model(self, state, config):
        if not config:
            raise ValueError('tensornet-ext loading requires model_config')
        model = self.build_model(config)
        model.load_state_dict(state, strict=True)
        return model
