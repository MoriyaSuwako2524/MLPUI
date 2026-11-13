"""TensorNet backend for excited-state transition charges / transition dipoles.

Dataset fields (NpyDataset):
  transition_charges  atomic layout with a trailing state axis: dense [S_frames, N, n_states]
                      or ragged [A, n_states]; [.., N] / [A] is accepted when n_states == 1
  transition_dipole   [S_frames, n_states, 3] (or [S_frames, 3] when n_states == 1), e*angstrom

Labels may carry an arbitrary sign per (frame, state). The Trainer calls
``align_targets`` per structure, which flips each state's label sign to the
exact minimiser of the weighted MSE (see mlpui.models.transition.phase), so
training loss and evaluation metrics are phase invariant.
"""
import copy
import math

import numpy as np
import torch

from .base import ModelBackend


class TransitionBackend(ModelBackend):
    name = 'tensornet_transition'
    aliases = ('transition',)
    label = 'TensorNet transition charges / dipoles'
    web_supported = False
    supports_evaluation_artifacts = False
    training_targets = ('transition_charges', 'transition_dipole')
    supports_batching = True

    def metadata(self):
        return {**super().metadata(), 'architecture': 'tensornet_transition_v1',
                'config_hint': 'Gas-phase TensorNet; per-state Sum q = 0 transition charges, '
                               'mu = Sum q r (+ optional atomic dipoles); phase-less (per-state sign) loss.'}

    def default_config(self):
        return dict(schema_version=1, architecture='tensornet_transition_v1', precision=32,
                    embedding_dimension=128, num_layers=2, num_rbf=32, max_num_neighbors=64,
                    cutoff_upper=5.0, num_states=1, atomic_dipoles=False,
                    units={'length': 'angstrom', 'charge': 'e', 'dipole': 'e*angstrom'})

    def _config(self, supplied):
        if (supplied.get('schema_version', 1) != 1
                or supplied.get('architecture', 'tensornet_transition_v1') != 'tensornet_transition_v1'):
            raise ValueError('Incompatible transition-property architecture')
        config = self.default_config()
        if set(supplied) - set(config):
            raise ValueError(f'Unknown transition config fields: {set(supplied) - set(config)}')
        config.update(copy.deepcopy(supplied))
        if config['precision'] not in (32, 64):
            raise ValueError('precision must be 32 or 64')
        for key in ('embedding_dimension', 'num_layers', 'num_rbf', 'max_num_neighbors', 'num_states'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        cutoff = config['cutoff_upper']
        if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('cutoff_upper must be finite and positive')
        if type(config['atomic_dipoles']) is not bool:
            raise ValueError('atomic_dipoles must be boolean')
        if config['units'] != {'length': 'angstrom', 'charge': 'e', 'dipole': 'e*angstrom'}:
            raise ValueError('Transition model requires angstrom / e / e*angstrom after dataset conversion')
        return config

    def configure_targets(self, config, targets):
        if set(targets) - set(self.training_targets):
            raise ValueError(f'Unsupported transition target: {set(targets) - set(self.training_targets)}')
        return self._config(config)

    def training_config(self, config, targets, dtype):
        config = self.configure_targets(config, targets)
        config['precision'] = 64 if dtype == torch.float64 else 32
        return config

    def add_charge_head(self, config):
        raise ValueError('Transition charges are the native output; use transition_charges, not charges')

    def matches(self, state):
        if isinstance(state, torch.nn.Module):
            return type(state).__module__ == 'mlpui.models.transition.model'
        return 'transition_schema' in state

    def build_model(self, config):
        from mlpui.models.transition.model import TransitionTensorNet
        return TransitionTensorNet(self._config(config))

    def load_model(self, state, config):
        if not config:
            raise ValueError('Transition model loading requires model_config')
        if 'transition_schema' not in state:
            raise ValueError('Not a transition-property checkpoint')
        model = self.build_model(config)
        model.load_state_dict(state, strict=True)
        return model

    def validate_training(self, model, config, targets):
        self.configure_targets(config, targets)

    def validate_checkpoint_config(self, saved_config, requested_config):
        if saved_config:
            saved, requested = self._config(saved_config), self._config(requested_config)
            saved.pop('precision')
            requested.pop('precision')
            if saved != requested:
                raise ValueError('Transition checkpoint config cannot be overridden')

    def available_properties(self, model):
        return set(self.training_targets)

    def validate_constraint(self, model):
        raise ValueError('Total-charge constraint does not apply: transition charges always sum to zero')

    # ---- inputs -------------------------------------------------------------
    def prepare_inputs(self, atoms, device, dtype, *, length_to_angstrom=1.,
                       charge=None, spin=0, require_charge=False, model=None):
        data, _, periodic = self.geometry(atoms, device, dtype, length_to_angstrom, charge, False)
        if periodic:
            raise ValueError('Transition model is nonperiodic')
        data.pop('batch')
        data['charge'] = torch.tensor([0. if charge is None else float(charge)], device=device, dtype=dtype)
        return data

    def prepare_sample(self, sample, device, dtype, *, model=None):
        if any(k in sample for k in ('mm_pos', 'mm_charges')):
            raise ValueError('Transition model v1 is gas phase; MM environment fields are not supported')
        if np.any(sample.get('pbc', False)):
            raise ValueError('Transition model is nonperiodic')
        z = np.asarray(sample['z'])
        pos = np.asarray(sample['pos'], dtype=np.float64)
        if z.ndim != 1 or len(z) == 0 or z.dtype.kind not in 'iu' or pos.shape != (len(z), 3):
            raise ValueError('Invalid transition sample shapes')
        inputs = {'z': torch.as_tensor(z, dtype=torch.long, device=device),
                  'pos': torch.tensor(pos, dtype=dtype, device=device)}
        batch = sample.get('batch')
        if batch is not None:
            inputs['batch'] = torch.as_tensor(np.asarray(batch), dtype=torch.long, device=device)
            count = int(inputs['batch'].max()) + 1
        else:
            count = 1
        charge = np.asarray(sample.get('charge', np.zeros(count)), dtype=np.float64).reshape(-1)
        if charge.shape != (count,) or not np.isfinite(charge).all():
            raise ValueError('Supply one finite total charge per structure')
        inputs['charge'] = torch.tensor(charge, dtype=dtype, device=device)
        return inputs

    def prepare_batch(self, samples, device, dtype, *, model=None):
        if any('batch' in s for s in samples):
            return None
        atoms = [len(np.asarray(s['z'])) for s in samples]
        merged = {'z': np.concatenate([np.asarray(s['z']) for s in samples]),
                  'pos': np.concatenate([np.asarray(s['pos']).reshape(-1, 3) for s in samples]),
                  'batch': np.repeat(np.arange(len(samples)), atoms),
                  'charge': np.array([float(np.asarray(s.get('charge', 0)).reshape(-1)[0]) for s in samples])}
        if any(np.any(s.get('pbc', False)) for s in samples):
            merged['pbc'] = True
        if any(k in s for s in samples for k in ('mm_pos', 'mm_charges')):
            merged['mm_pos'] = True   # rejected by prepare_sample with the gas-phase message
        return self.prepare_sample(merged, device, dtype, model=model), {'atoms': atoms}

    def forward(self, model, inputs, properties):
        if set(properties) - set(self.training_targets):
            raise ValueError('Unsupported transition property')
        return model(**inputs, properties=properties)

    # ---- phase --------------------------------------------------------------
    def align_targets(self, model, predictions, targets, weights):
        """Per-structure, per-state label sign that minimises the weighted MSE."""
        from mlpui.models.transition.phase import align_transition_targets
        return align_transition_targets(model.num_states, predictions, targets, weights)
