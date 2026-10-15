"""TensorNet-QMMM backend with explicit baseline provenance."""
import copy
import math
import re
import numpy as np
import torch
from .base import ModelBackend


class TensorNetQMMMBackend(ModelBackend):
    name = 'tensornet_qmmm'
    label = 'TensorNet-QMMM'
    web_supported = True
    supports_evaluation_artifacts = True
    training_targets = ('energy', 'forces', 'mm_forces', 'mm_potential', 'mm_potential_gradient')

    def default_config(self):
        return dict(schema_version=1, precision=32, embedding_dimension=64, num_layers=3,
            num_rbf=32, max_num_neighbors=64, cutoff_upper=5., environment_cutoff=10.,
            environment_mode='fixed_reference_selection', potential_scale=1., field_scale=1.,
            units={'length': 'angstrom', 'energy': 'eV', 'charge': 'e'},
            baseline={'mode': 'residual_only'})

    def _config(self, supplied):
        config = self.default_config()
        if set(supplied) - set(config):
            raise ValueError(f'Unknown QMMM config fields: {set(supplied) - set(config)}')
        config.update(copy.deepcopy(supplied))
        if type(config['schema_version']) is not int or config['schema_version'] != 1:
            raise ValueError('Unsupported QMMM schema_version')
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

    def configure_targets(self, config, targets):
        if set(targets) - set(self.training_targets):
            raise ValueError('Unsupported QMMM target')
        config = self._config(config)
        self._check_force(config, targets)
        return config

    def _check_force(self, config, targets):
        baseline = config['baseline']
        if 'forces' in targets and baseline.get('qm_force_kind') == 'fixed_charge_approximation':
            raise ValueError('Complete QM forces require dq/dR; precomputed fixed charges are insufficient')

    def training_config(self, config, targets, dtype):
        config = self.configure_targets(config, targets)
        config['precision'] = 64 if dtype == torch.float64 else 32
        return config

    def add_charge_head(self, config):
        raise ValueError('Gas charge baseline is separate; QMMM does not train a charges head')

    def matches(self, state):
        if isinstance(state, torch.nn.Module):
            return type(state).__module__ == 'mlpui.models.tensornet_qmmm.model'
        return 'qmmm_schema' in state

    def build_model(self, config):
        from mlpui.models.tensornet_qmmm.model import TensorNetQMMM
        return TensorNetQMMM(self._config(config))

    def load_model(self, state, config):
        if not config:
            raise ValueError('QMMM loading requires model_config')
        if 'qmmm_schema' not in state or state['qmmm_schema'].numel() != 1 or state['qmmm_schema'].item() != 1:
            raise ValueError('Unsupported QMMM weight schema')
        model = self.build_model(config)
        model.load_state_dict(state, strict=True)
        return model

    def validate_training(self, model, config, targets):
        self.configure_targets(config, targets)

    def validate_checkpoint_config(self, saved_config, requested_config):
        if saved_config:
            saved, requested = self._config(saved_config), self._config(requested_config)
            # Precision conversion is allowed; physical definitions cannot change.
            saved.pop('precision')
            requested.pop('precision')
            if saved != requested:
                raise ValueError('QMMM checkpoint config/provenance cannot be overridden')

    def available_properties(self, model):
        return set(self.training_targets)

    def prepare_inputs(self, atoms, device, dtype, **kwargs):
        raise ValueError('QMMM needs a structured dual-region sample; ASE/MD adapter is not implemented')

    def prepare_sample(self, sample, device, dtype, *, model=None):
        if model is None:
            raise ValueError('QMMM prepare_sample requires model configuration')
        if np.any(sample.get('pbc', False)):
            raise ValueError('First-release QMMM is nonperiodic')
        required = {'z', 'pos', 'mm_pos', 'mm_charges'}
        if not required <= sample.keys():
            raise ValueError(f'Missing QMMM inputs: {required - sample.keys()}')
        inputs = {k: torch.tensor(np.asarray(sample[k]), device=device,
                  dtype=torch.long if k == 'z' else dtype) for k in required}
        z, pos, mp, mq = (inputs[k] for k in ('z', 'pos', 'mm_pos', 'mm_charges'))
        if (z.ndim != 1 or len(z) == 0 or pos.shape != (len(z), 3) or mp.ndim != 2
                or mp.shape[-1] != 3 or mq.shape != (len(mp),)):
            raise ValueError('Invalid dual-region sample shapes')
        if np.asarray(sample['z']).dtype.kind not in 'iu' or torch.any((z < 1) | (z > 118)):
            raise ValueError('QM atomic numbers must be integers 1..118')
        if not all(torch.isfinite(v).all() for v in inputs.values()):
            raise ValueError('Non-finite QMMM input')
        inputs['charge'] = torch.tensor(float(np.asarray(sample.get('charge', 0)).item()), device=device, dtype=dtype)
        if not torch.isfinite(inputs['charge']):
            raise ValueError('QM charge must be finite')
        if float(np.asarray(sample.get('spin', 0)).item()) != 0:
            raise ValueError('QMMM residual currently supports spin=0 only')
        baseline = {}
        shapes = {'energy': ((), (1,)), 'forces': (tuple(pos.shape),),
                  'mm_forces': (tuple(mp.shape),), 'mm_potential': (tuple(mq.shape),),
                  'mm_potential_gradient': (tuple(mp.shape),)}
        for prop, allowed in shapes.items():
            key = 'baseline_' + prop
            if key in sample:
                value = torch.tensor(np.asarray(sample[key]), device=device, dtype=dtype)
                if tuple(value.shape) not in allowed or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid {key}')
                baseline[key] = value.reshape(()) if prop == 'energy' else value
        if baseline and model.config['baseline']['mode'] != 'offline':
            raise ValueError('Baseline arrays require explicit offline mode and provenance')
        if model.config['baseline']['mode'] == 'offline':
            if sample.get('baseline_metadata') != model.config['baseline']:
                raise ValueError('Dataset baseline_metadata must exactly match checkpoint provenance')
        inputs['baseline'] = baseline
        return inputs

    def forward(self, model, inputs, properties):
        if set(properties) - set(self.training_targets):
            raise ValueError('Unsupported QMMM property')
        self._check_force(model.config, properties)
        return model(**inputs, properties=properties)
