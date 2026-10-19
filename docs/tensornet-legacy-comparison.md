# Comparing the historical model with TensorNet-QMMM

The Web model selector now offers two independent options. The user-facing
name `tensornet-ext` deliberately denotes the **old MLPUI model**, not the
upstream panxl TensorNet-ext implementation underpinning current QMMM.

| Web / family | Architecture | State schema | Residual computation |
| --- | --- | --- | --- |
| TensorNet-QMMM / `tensornet_qmmm` | `tensornet_ext_v1` | 2 | Radial MM tensor messages injected into TensorNet interactions |
| tensornet-ext (legacy QM-only + H+3) / `tensornet-ext` | `tensornet_qm_h3_v1` | 1 | QM-only TensorNet h; concatenate phi, squared field magnitude, field dot geometric vector; H+3 -> H -> SiLU -> 1, summed over QM atoms |

The legacy computational implementation is recovered from commit
`f7536676e02e1a64d1f0c9be9042e27b24d836db`, before replacement in
`7c3ff37d489cd8137de9226990ea2b5775b6e270`. The unchanged historical module is
retained as `tests/fixtures/qmmm_schema1_f753667.py` for numerical regression.
The production copy adds architecture guards and an independent-frame ragged
batch adapter. It does not introduce a new approximation to the old network.
The current QMMM neural implementation is unchanged.

## Training a fair comparison

Use `configs/train_tensornet_ext.yaml` for the old architecture, or select
`tensornet-ext (legacy QM-only + H+3)` in the Web UI. The Python registry also
accepts `tensornet_ext` as an alias; saved metadata uses `tensornet-ext`.
Both expose energy, QM forces, MM potential and fixed-response GESP weights:
`energy`, `forces`, `mm_potential`, `mm_potential_gradient`. Optional MM force
supervision is also retained. The same trainer, evaluation dispatch, optimizer,
seed, batching, precision, checkpoint settings and per-frame normalization apply.
Changing the model does not alter dataset splits, unit conversion or loss weights.

Choose identical train/validation/test records, seed, epochs, batch size, learning
rate, precision, shared network dimensions/cutoff, baseline provenance and four
weights for the comparison. The legacy preset retains its historical defaults
(H=64, 3 layers, QM cutoff=5 A), while QMMM retains H=128, 2 layers, cutoff=4.5 A;
these differences are visible and should be set explicitly for a controlled run.
The supplied legacy comparison YAML matches the current QMMM YAML's shared
network/training settings but enables all four targets; enable the same weights
in the QMMM configuration. Parameter counts still differ between architectures.
Legacy-only `potential_scale` and `field_scale` default to 1; QMMM's `ext_*`
radial settings do not apply to the old model and are rejected there.

Both consume the existing dual-region datasets, including independent ragged
QM/MM offsets and empty MM regions. Length/energy/charge units are angstrom/eV/e.
Forces are -dE/dR, MM potential is +dE/dq_MM, and GESP is the fixed-response
spatial gradient, excluding response Hessian terms and requiring no division by
MM charge. `residual_only` expects already-subtracted labels. `offline` adds the
supplied UMA gas + gas-charge Coulomb baseline for every requested target and
requires identical provenance, including the full dq/dR force convention.
Full-result evaluation exports preserve units, frame IDs and ragged MM layout.
See [the baseline contract](tensornet-qmmm.md) for dataset details.

## Checkpoints

New checkpoints identify backend and architecture separately. Schema2 weights
cannot load into the old model, and schema1 weights cannot load into current
QMMM, including direct non-strict state loads. Architecture/config overrides
and baseline provenance mismatches are rejected.

Historical **state-dict** checkpoints labeled `backend: tensornet_qmmm` can be
loaded by explicitly requesting `family='tensornet-ext'`; their saved config must
have schema_version 1 and no architecture (or `tensornet_qm_h3_v1`). The loader
checks the schema marker and strictly checks all weight names/shapes. Auto-load
of those old labeled files is intentionally rejected. Bare schema1 state dicts
can be detected as the legacy option but require the exact original model config.
Historical full-module pickles at the old QMMM class path remain rejected; export
a state dict in the original environment. New full-module checkpoints retain the
existing explicit trusted-checkpoint requirement.

No training service is restarted and no production training is started by adding
this option. The tests use tiny synthetic data and isolated temporary servers.

## Validation

Validated on Windows with PyTorch 2.11.0, CPU float64 and
`MLPUI_NEIGHBORS=python`:

- 66 focused Python checks across `test_tensornet_legacy.py`,
  `test_tensornet_ext.py`, `test_qmmm.py`, `test_qmmm_gesp.py`,
  `test_qmmm_review.py` and `test_backends.py`.
- The bundled-backend isolated training test passes with the new option.
- Eight HTTP end-to-end combinations pass (both architectures, residual/offline,
  with/without GESP), plus an HTTP presets identity check. These use synthetic
  data, one epoch and temporary localhost servers, then reload checkpoints and
  compare full evaluation exports.
- 15 frontend checks pass with `node --test tests/web_ui.test.cjs`.

The historical source SHA256 after LF normalization is
`55d111d9e0b878ec46860a82bfb54f1534dbae481782e58db3774ea92a2f0931`.
The regression requires exact equality of all four outputs and their parameter
gradients, including zero-MM and zero-charge cases. Ragged/interleaved batching,
full baseline reconstruction and wrong-architecture rejection are tested.
CUDA/native-neighbor performance and real-data training are not validated here.
