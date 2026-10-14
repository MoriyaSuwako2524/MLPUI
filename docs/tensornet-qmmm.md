# TensorNet-QMMM: fixed-environment prototype

This is a new TensorNet adaptation, not a reproduction of the Pan DeepPot-SE
architecture. It uses the bundled
`mlpui.models.torchmdnet.models.tensornet.TensorNet`; the old standalone TensorNet
directory is unchanged. There is no QM/MM-AC, periodic embedding or MD adapter.

## Energy and baseline semantics

The intended total embedded-QM energy is

```
E = E_UMA_gas + Coulomb(q_gas(R_QM), Q_MM) + E_residual
```

The residual includes both the gas-level correction and environment residual.
Classical MM internal energy, MM self energy, LJ and other coupling terms are
not silently included or subtracted. Establish the label energy definition
before using real data. ESP-fitted atomic charges are not the same quantity as
the MM potential label `+dE/dQ_MM`.

Web, CLI and Python support two explicit modes:

* `baseline: {mode: residual_only}`: output and labels are residual energy and
  its derivatives. Pre-subtract a consistent baseline from reference labels.
  No UMA or charge model is loaded and the output is not a completed total
  QM/MM potential. This mode also provides a standalone numerical prototype.
* `baseline.mode: offline`: output is supplied gas-plus-Coulomb baseline plus
  learned residual. Supply `baseline_energy` and the corresponding baseline
  derivative arrays for each enabled target. Energy labels are total embedded
  QM reference energies under the chosen convention. Baseline arrays are
  externally computed values at the exact sample, not an interpolatable energy
  function; finite-difference checking them requires regenerating the baseline
  at the perturbed geometry. They cannot be used for MD.
  Outputs explicitly report `derivative_contract:
  offline_component_values_not_total_energy_autograd`. Autograd of the returned
  energy only sees the residual coordinate dependence; use the separately
  composed force/potential outputs for total sample derivatives. Energy-loss
  gradients with respect to residual parameters remain valid.

For offline mode, include this object as `model_config.baseline` and as the
complete contents of `baseline_metadata.json` in every dataset directory:

```json
{
  "mode": "offline",
  "energy_definition": "uma_gas_plus_gas_charge_coulomb",
  "qm_force_kind": "full_dq_dR",
  "uma_checkpoint_sha256": "<64 lowercase hexadecimal characters>",
  "charge_checkpoint_sha256": "<64 lowercase hexadecimal characters>",
  "source_manifest_sha256": "<64 lowercase hexadecimal characters>"
}
```

Use actual digests, not the placeholders. The source manifest should record
frame pairing, units/conversions, label level, environment selection, baseline
generation and the MM-self-energy convention. This release validates metadata
shape and exact dataset/checkpoint agreement, not the truth of the supplied
provenance or the contents of externally stored source files. Source manifests
for separate shards must share the identity of the common dataset manifest.

`qm_force_kind: fixed_charge_approximation` permits energy/MM-derivative work
but is rejected when QM forces are requested. Precomputed gas charges alone
do not provide dq/dR and cannot establish complete QM baseline forces.

`GasCoulombBaseline` is a separate Python composition hook for deterministic
differentiable Torch energy and charge providers. It freezes provider parameters
and keeps them in eval mode without detaching their coordinate graphs. It
explicitly projects gas charges as `q = q_raw - mean(q_raw) + Q_QM/N`. Thus the
sum is Q_QM and its coordinate Jacobian sums to zero; projection and dq/dR are
both retained in autograd. Matching offline baselines must use this same charge
convention, recorded in their source manifest. This is an explicit baseline
definition, not a claim that arbitrary provider predictions already conserve
charge. Its
`compose(residual_model, inputs)` returns total energy and gas/Coulomb/residual
components; differentiate this total for QM/MM forces and MM potential. There
is no automatic online UMA/charge-weight loader or portable serialization for
this hook yet. Synthetic tests exercise the derivative contract; they do not
validate a specific UMA checkpoint. Never replace a real provider with test
functions for scientific training.

## Data contract

All files are numeric NPY without pickle. Units after conversion are angstrom,
eV and elementary charge. The Coulomb coefficient is 14.3996454784255 eV A/e^2.
Use `energy_scale` and `length_scale` to convert raw arrays; charges are already
in e. Fixed environments are selected upstream within 10 A of a reference QM
geometry. All supplied MM points are used, with no runtime distance filtering,
switching, reselection or claims of MD cutoff continuity.

| Field | Dense layout | Ragged layout |
|---|---|---|
| `z` | `[N]` or `[S,N]` | `[A]` |
| `pos`, `forces`, `baseline_forces` | `[S,N,3]` | `[A,3]` |
| `offsets` | omitted | `[S+1]`, strictly increasing |
| `mm_pos`, `mm_forces`, `baseline_mm_forces` | `[S,M,3]` | `[B,3]` |
| `mm_charges`, `mm_potential`, `baseline_mm_potential` | `[S,M]` | `[B]` |
| `mm_offsets` | omitted | `[S+1]`, nondecreasing; zero-MM samples allowed |
| `energy`, `baseline_energy` | `[S]` or `[S,1]` | same |

The two layouts are independent: ragged QM can accompany dense MM and vice
versa. No padding/dummy elements. QM must contain at least one atom. The data
loader supports optional per-structure total QM `charge`; absent means zero.
Nonzero spin and PBC are rejected by this prototype.

`forces = -dE/dR_QM`, `mm_forces = -dE/dR_MM`, and
`mm_potential = +dE/dQ_MM`. Potential units are eV/e. `gradients: true` negates
reference QM/MM gradient labels into forces. Baseline force files must already
have the force sign and are never negated by this option. Coordinate/force/
potential scaling applies to both regions and baseline arrays. Four targets
are independently selectable through `training.loss_weights`. Empty MM targets
contribute zero training loss; numeric evaluation reports count=0 and null
metrics when no MM components exist. Target selection is dataset-wide: every
sample must supply every enabled label. There are no per-atom/component masks or
partially missing frame labels in this version; NaN placeholders are rejected.
An omitted target contributes no loss, while zero-MM is a physically empty
region, not a missing-label mask. Per-structure loss averaging includes these
zero contributions, so it differs from averaging only MM-labeled components.

For potential supervision in residual-only mode, subtract
`baseline_mm_potential = +d(Egas + ECoul)/dQ_MM` from the reference potential.
With gas providers independent of Q_MM this is `k * sum_i(q_i/r_iM)`; it is
not the negative of that quantity and not an atomic ESP charge. Offline mode
adds this baseline to the prediction, so its error is algebraically the same
as comparing residual prediction against (reference - baseline). If an energy
label has MM self energy removed, remove the corresponding self-energy
derivatives from MM force/potential reference labels as well.

## Architecture and differentiation

TensorNet produces invariant QM features. The environment branch computes
potential phi and polar field E differentiably from QM/MM coordinates and MM
charges. A residual atomic scalar head consumes the QM features and three
O(3) invariants: phi, E.E, and E.g, where g is a smoothly weighted sum of QM
relative vectors. Atomic energies are summed. Translation, rotation,
reflection and independent atom/point permutations are tested. This is a
minimal descriptor, not a claim of the paper's expressive power or accuracy.
The single E.g projection loses directional information: fields with identical
norm and projection on g are indistinguishable to that atomic head. At g=0 it
retains only field magnitude. Symmetry correctness does not imply angular
completeness; anisotropic response may require additional independent geometric
channels or equivariant tensor fusion in a later architecture version.

Forces and potential are differentiated from the same residual scalar.
Training retains higher derivatives for force/potential losses. Coincident
QM/MM sites (distance below 1e-6 A) are rejected rather than silently softened.
Descriptor computation uses dense QM-MM pairs; large environment performance
is not established. The existing TensorNet neighbor backend may use its
quadratic PyTorch fallback when compiled operators are absent.

## Training and loading

Start from `configs/train_tensornet_qmmm.yaml`, fill paths and supply labels
consistent with the selected baseline mode:

```powershell
.venv\Scripts\python.exe -m scripts.train configs/train_tensornet_qmmm.yaml
```

Python: `Trainer('tensornet_qmmm', model_config, TrainingConfig(...))` accepts
`NpyDataset`/`NpyShards`. For inference, use the backend's `prepare_sample()` and
`forward()` with a model loaded by `load_external_checkpoint()`. Model `forward`
is single-structure; the existing trainer accumulates per-structure gradients
for batches. Raw `model.energy_components()` only returns the residual.

Checkpoints store complete configuration, baseline metadata and strict state
dict under a unique backend and weight-schema marker. Loading rejects missing
or unexpected weights, unsupported schemas, and changes to physical config or
baseline identities. Precision conversion is permitted. As with other Trainer
exports, optimizer/RNG state is not saved: continuation is not exact resume.
Only residual model weights are embedded. UMA and gas-charge provider weights,
offline baseline arrays and the source manifest itself are NOT embedded: their
digests are identities, not downloadable dependencies. Loading an offline
checkpoint without matching per-sample baseline data fails on prediction. A
roundtrip proves residual/config reproducibility with the same supplied
baseline; it does not establish a self-contained total-energy MD potential.

## Deliberate first-release boundaries

No real Pete data has been paired, converted or trained here. Gas charge
prediction arrays are not charge-model weights. Real-data use still needs
frame matching, unit, gas-model identity and embedded energy/self-energy checks.

Web training and evaluation support dual-region uploads/imports, splitting and
parity/artifact export. ASE/MD remains unsupported. MD also requires online providers, environment/cache
updates, smooth dynamic selection, full force composition and conservation tests.

Acceptance tests: `python -m pytest tests/test_qmmm.py -q`. They cover ragged and
empty regions, O(3), coordinate/charge finite differences, derivative-loss
parameter gradients, frozen-charge-provider dq/dR, actual CLI training, strict
checkpoint roundtrip, provenance guards and Web upload/split/train/evaluate/download/reload flows.


## Web workflow

1. Open Dataset Library and upload numeric NPY files plus optional
   `baseline_metadata.json`, or import a server directory. Use the QMMM mapping
   preset, then edit the mapping for the actual filenames. Omit `offsets` or
   `mm_offsets` for a dense region; include each independently for ragged data.
   Add optional MM targets, baseline arrays and `frame_index`, `source_index`,
   `short_contact` mappings when present. Offline metadata can also be supplied
   in the dataset metadata JSON field. Labels must match the selected targets.
2. Inspect the dataset, then split using the desired seed/ratios or ordered mode.
   Export preserves original frame IDs, independent region offsets, baseline
   values/identity and supplied flags. Unit conversion is applied once; exported
   scales are reset to one. Ratio splitting does not implement scientific
   per-window guard intervals: import precomputed splits for those protocols.
3. Select TensorNet-QMMM and use the train split in the job. Managed sibling
   validation/test splits are selected automatically. Choose residual_only or
   offline; managed offline datasets populate their identity metadata. For manual
   paths, use the same offline identity as the sidecar in every dataset directory.
4. Set positive loss weights only for labels you have (zero disables a target).
   Energy, QM forces, MM forces and MM potential are independent. Use Check data
   before Start. Missing selected labels, missing baseline arrays and mismatched
   baseline identities are also rejected by the server before queuing a worker.
5. Download model.pt when training finishes. Switch to Evaluation, select the
   matching test dataset/checkpoint and targets. Download reference/prediction
   NPY arrays and metadata.json from the result. The manifest records units,
   region/layout, counts, offsets, frame identity and baseline derivative contract.
   Zero-MM evaluations report count=0 and N/A metrics and omit empty parity plots.
6. Reload the checkpoint with `load_external_checkpoint(..., device="cpu")`, then
   pass matching structured samples to backend.prepare_sample()/forward(). Offline
   checkpoints still require the external baseline arrays for each sample.

`tests/test_web_qmmm.py` uses synthetic six-frame dual-ragged data with zero-MM
frames, both baseline modes, isolated random-port servers and real worker
processes. It compares downloaded predictions with the downloaded/reloaded model.
No real-data training or online UMA/charge provider integration is implied.
