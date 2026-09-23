# Model backends

`mlpui.backends` is the internal registry used by the trainer, checkpoint loader,
ASE adapters and WebUI. It registers NewtonNet, TorchMD-Net (which includes
TensorNet) and MACE. Model implementations remain under `mlpui.models`; this interface does
not change their parameters, equations or native state-dict keys.

## Responsibilities

Each stateless `ModelBackend` owns:

- Default architecture settings, aliases, display metadata and target configuration.
- Model creation and strict restoration of native weights, including legacy migrations.
- Conversion of ASE structures to its native input representation.
- Forward execution, gradient context and canonical property outputs (`energy`,
  `forces`, `charges`, `dipole`, `stress`, and the ASE alias `free_energy`).
- Training validation and the properties available on an actual model instance.

The shared trainer owns NPY datasets, weighted losses, optimizer updates,
validation/testing, early stopping, progress and checkpoint scheduling. The shared
calculator owns units, ASE results, patching and device/offload behavior. The
WebUI gets its model list, names, configuration nesting and target options from
the registry via `/api/presets`; the existing `models` response field is retained.

Backend `training_targets` describes potential supervised properties; it does not
promise that a particular checkpoint contains those trained heads. Inference uses
`available_properties(model)`. Missing learned heads are never silently created
while loading a checkpoint. Model imports occur only when a model is constructed
or loaded; registry discovery does not import the native model implementations.

## Adding a backend

Implement `ModelBackend` in a new module and register one instance in
`mlpui/backends/__init__.py`. Names and aliases must be unique. Keep per-model
state on the model, never on the registered adapter. `prepare_inputs` may return
any native input object; `forward` handles that object's API and returns a dict
with canonical property names. Training and ASE pass the active model to
`prepare_inputs`, so graph construction can use its element table and cutoff.
Force training must preserve the differentiation
graph. Evaluation may still require coordinate gradients even though parameters
are not updated. A future graph backend can build its own neighbor data here.

Use `geometry()` only if the existing nonperiodic/fully-periodic validation is
appropriate. Each backend remains responsible for its own cell restrictions.
Preserve user configurations by copying before modifications. Respect energy
head ordering and distinguish predicted atomic charges from input total charge.
NewtonNet continues appending supervised charge heads after existing heads; an
explicit charge-before-energy configuration continues using LES. All three backends
retain their optional differentiable total-charge constraint.

`tests/test_backends.py` includes a third backend with a different input API to
exercise registry dispatch through training, evaluation, WebUI configuration,
checkpoint loading and ASE without adding family branches to those components.
The existing numerical, force, charge, legacy checkpoint and worker tests remain
the compatibility suite. New backends also need their own numerical tests.

## Checkpoint compatibility

New trainer exports add `backend`, `format_version: 1` and
`model_config_version: 1` alongside the existing `model_config`, `state_dict`,
history, epoch and early-stopping fields. The architecture version belongs to the
backend; the file format version belongs to the shared loader. Unsupported
versions and conflicting backend identities fail explicitly. Old files without
these fields continue using native weight/module identification and the existing
normalization and key migrations. Exports still do not contain optimizer/RNG
state for exact training resume.

The registry does not change serialization trust: full Python-object checkpoints
still require explicit `trusted_checkpoint=True`; normal exports use weights-only
loading. `models/checkpoint_pickle.py` retains the historical module-name mapping
for the two vendored packages. UMA and the legacy simple-calculator fallback are
outside this registry and retain their existing paths. See [MACE](mace.md) for
the bundled MACE backend, its supervised-charge extension and supported scope.
