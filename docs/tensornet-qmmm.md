# TensorNet-QMMM: TensorNet-ext residual architecture

Schema 2 (`architecture: tensornet_ext_v1`) replaces the original H+3 head.
It follows [panxl/torchmd-net at b0f7c7b](https://github.com/panxl/torchmd-net/tree/b0f7c7b42c6304c016a9e3dee78d4e2f4cbadc7d).
Legacy schema-1 checkpoints are rejected with an architecture-incompatibility
error, including direct non-strict state loading. Existing files are not changed.
The historical model is separately available as [`tensornet-ext`](tensornet-legacy-comparison.md); it does not change this schema2 implementation.
This is a finite, nonperiodic residual adaptation; there is no QM/MM-AC or MD adapter.

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
geometry. This **data selection** remains fixed. The neural external message
separately uses `ext_cutoff_upper` (default 10 A) with the upstream cosine
cutoff and RBF envelope; pairs outside it contribute zero neural residual.
The gas-charge Coulomb baseline still uses all supplied MM points, without
this neural cutoff. No periodic minimum image, reselection or softening is added.

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

For each within-frame QM-MM edge, r=R_QM-R_MM, u=r/|r|. The external
radial basis passes through K->H->2H->3H linear layers with SiLU after **each**.
After multiplying by q_MM and the cosine cutoff, its three channels weight
I, skew(u), and u outer u - I/3. Their sum aggregates into [N_QM,H,3,3].
This message is computed once and added to the QM message in every Interaction,
before the upstream O(3) tensor update. There is no MM-MM message passing.
The upstream invariant readout and Scalar head H->H/2->1 produce atomic
energies, summed per structure. The old phi/|E|^2/E.g head is removed.
The nonlinear interactions allow charge/environment response despite the
charge-linear external message. Total QM charge is retained for baseline/data
contracts; upstream ext does not condition its residual on total QM charge.
Train a consistent charge state; this residual does not distinguish electronic
charge states with identical geometry and external charges.

Defaults follow the upstream **class**: H=128, two layers, QM cutoff=4.5 A,
32 QM and 32 external expnorm RBFs, external cutoff=10 A, SiLU, O(3),
non-trainable RBFs. Upstream CLI uses a conflicting 5 A external cutoff;
MLPUI explicitly chooses 10 A, shared by Python/Web/YAML. External cutoff,
RBF count/type/trainability and lower bound are serialized and configurable.
Existing loss weights, split handling and short-contact flags are unchanged.

Upstream adaptations: explicit ragged batch/offset indexing replaces fixed-count
reshape; index_add with N_QM size preserves isolated/trailing atoms and empty MM
regions. Coincident QM/MM sites (<1e-6 A) raise rather than being softened.
Per-frame dense pair construction avoids cross-frame pairs. The vendored QM
neighbor backend may use its quadratic CPU fallback. No upstream global TF32
settings or ESP/GESP EMA training code is imported; MLPUI's existing independent
target losses remain in use, avoiding upstream esp_grad/esp EMA-key coupling.
The adapted interactions and pinned test fixture retain MIT attribution/license.

## Training and loading

Start from `configs/train_tensornet_qmmm.yaml`, fill paths and supply labels
consistent with the selected baseline mode:

```powershell
.venv\Scripts\python.exe -m scripts.train configs/train_tensornet_qmmm.yaml
```

Python: `Trainer('tensornet_qmmm', model_config, TrainingConfig(...))` accepts
`NpyDataset`/`NpyShards`. For inference, use the backend's `prepare_sample()` and
`forward()` with a model loaded by `load_external_checkpoint()`. Model `forward`
accepts a single structure or concatenated regions with QM `batch` and either
MM `mm_batch` or nondecreasing `mm_offsets`; batched total `charge` has shape [B].
The backend accepts the same batch metadata and shape-checked offline arrays.
The existing trainer still accumulates per-structure gradients for batches.
Raw `model.energy_components()` only returns the residual.

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

Architecture unit tests do not establish trained accuracy. Gas charge
prediction arrays are not charge-model weights. Real-data use requires
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

## Fixed-response potential gradient

The optional `mm_potential_gradient` target is GESP in eV/(e Angstrom),
with shape [total MM points,3] for ragged data. Its sign is the spatial
gradient of potential, opposite to electric field. `gradients=True` only
changes force-label signs; it does not negate this target. Unit conversion
is energy_scale/length_scale. Offline mode additionally requires
`baseline_mm_potential_gradient` with the same shape and units.

Write the shared external message m_i=sum_a q_a K_ia(R) and
c_i=dE/dm_i, including every layer that reuses m. ESP is dE/dq_a=sum_i c_i:K_ia.
GESP is the **fixed-response** derivative sum_i c_i:dK_ia/dR_a. This equals
(dE/dR_a)/q_a for nonzero q_a, with the upstream gradient sign (not force).
It is not d(ESP)/dR_a, which also differentiates c and includes response terms.

The implementation first computes c, then creates an independent probe node
Rprobe=R_MM.clone() and evaluates unit-charge kernels at that probe. Differentiating
sum_i c_i:sum_a K_ia(Rprobe) with respect to Rprobe holds c fixed spatially.
It includes radial basis, cosine cutoff and direction derivatives. c is **not**
detached, so GESP-loss backpropagation includes dc/dtheta as well as kernel-weight
derivatives. No division by charge or epsilon substitution occurs; zero-charge
sites have the finite continuous response-kernel limit. The same algorithm
supports empty MM regions and retains training graphs.

QM forces remain the full -dE/dR_QM, ESP is +dE/dq_MM, and MM forces are
-dE/dR_MM. The chain rule gives F_MM=-q_MM*GESP for this residual architecture.
This defines the learned fixed response, not an explicit electronic density.
These derivative identities refer to the residual model. In `offline` mode,
separately supplied baseline values are added to each output; their physical
consistency cannot be established by differentiating the supplied scalar energy
array. The existing offline derivative contract still applies, and arbitrary
baseline arrays need not satisfy F_MM=-q_MM*GESP. The CM preparation script
produces residual-only labels and requires the original baseline provenance.

The implementation is in `mlpui/models/tensornet_qmmm/model.py`; numeric tests
are in `tests/test_qmmm_gesp.py` and `tests/test_tensornet_ext.py`. Tests check
full-energy coordinate/charge
finite differences, unchanged E/F/ESP when GESP is requested, frozen-response
probe finite differences, an explicit counterexample to the total mixed
derivative, zero/small MM charges, O(3) transformations, each target's parameter
backward, frame normalization, and checkpoint reproduction.

For residual CM targets, subtract the same gas-charge Coulomb potential and
fixed-charge spatial gradient used by the E/F baseline. UMA gas energy has
no MM derivative. QM force baselines must retain dq/dR; that requirement
does not introduce dq/dR_MM into the fixed-charge spatial GESP baseline.
Match checkpoint identity, projection, Coulomb constant, geometry and frame
IDs before constructing residual labels. Do not replace E/F residuals.

Training and validation loss use a component mean within each frame, then
an equal frame mean (empty MM contributes zero). Standalone evaluation
instead pools scalar components and reports its component count. Initial
four-target weights should be inverse squared train-only RMS scales,
computed with the same per-frame convention and recorded with their units.
Validation selects stopping; test/guard frames must not set scales or be
attached to the training job. Schema-2 E/F target selections remain valid;
schema-1 model configurations must be replaced.

`python -m scripts.prepare_cm_fourtarget --help` documents the CM importer.
Its inputs are the original mm10A/npys directory, verified ESP npys directory,
downloaded gas prediction artifacts, original residual E/F train/validation
directories, a new output directory and a reviewed baseline contract JSON.
The contract requires `gas_charge_checkpoint_sha256`, `gas_prediction_sha256`,
`coulomb_constant`, `charge_projection` (`as_saved` or `uniform_total_charge`),
`qm_force_includes_dq_dR: true`, `mm_self_excluded: true`, and
`ef_source_sha256` mapping train/validation to energy/forces file hashes.
Obtain this contract from the original preparation code and artifacts; the
importer cannot establish its truth from the E/F residual arrays alone.
It verifies the fixed split, atomic numbers, geometry, MM charges and all ESP
export hashes, copies E/F bytes unchanged, and writes only train/validation.
It refuses an existing output directory. Use the resulting `scales.json`
loss_weights in a new run with the original architecture and seed, attaching
validation only. `preparation.json` records the complete input identities.
