# UMA prediction in WebUI

Choose **New job > Job type > UMA prediction**. Select a local UMA checkpoint
on the WebUI server and an NPY dataset. The job uses the existing CPU/GPU queue
and produces energy and forces without reference labels.

## Runtime and supported models

The WebUI uses MLPUI's complete native UMA predictor, including output heads,
normalizers and elemental energy offsets. No FAIR-Chem installation, separate
Python environment, model download or remote service is required.

The validated checkpoint is **uma-s-1p2.pt**, task **omol**, for nonperiodic
molecules. Unsupported checkpoint heads fail explicitly; the older uma-s-1p1.pt
head format is not supported. UMA training, periodic systems, stress and atomic
charge prediction are not offered by this task. See [native UMA setup](uma.md).

## Input data

- Required: `z.npy` and `pos.npy`. Custom filenames and `{shard}` groups work.
- Optional: `cell.npy`, `pbc.npy`, `charge.npy`, `spin.npy`, `offsets.npy`.
  PBC must be false. WebUI structures must contain at least two atoms; isolated
  atomic references are available through the Python API, not this form.
- For variable atom counts, use flattened positions and atomic numbers plus offsets.
- Defaults: `{"task":"omol","charge":0,"spin":1}`. Charge is total molecular Q;
  spin is multiplicity (1 for a singlet), not the number of unpaired electrons.
- Optional per-structure charge/spin NPY arrays override these defaults. Leave
  mappings blank to use the defaults. Charge must be an integer in [-100,100];
  spin must be an integer in [1,100].
- Convert positions and cell to angstrom using **Length scale**. Prediction outputs
  are always eV and eV/angstrom. Label energy scaling does not rescale predictions.
- The WebUI uses float32 and computes both energy and forces.

## Results

The completed job offers downloads for:

- `energy.npy`: one energy per structure, shape `[S]`, in eV.
- `forces.npy`: concatenated atom forces, shape `[A,3]`, in eV/angstrom.
- `offsets.npy`: integer structure boundaries, shape `[S+1]`.
- `prediction.json`: layout, units and counts.

For structure `i`, use `forces[offsets[i]:offsets[i+1]]`. Input order is retained,
including group order. Cancelled or failed jobs do not expose results as complete.
Prediction datasets participate in managed-file reference protection.
