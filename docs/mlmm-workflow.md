# ML/MM end to end: from an Amber `.nc` trajectory to a new ML/MM trajectory

This is the operational guide for the full ML/MM loop:

```
QM/MM umbrella trajectory (.nc)
  -> 1. replay with sander + QMHub: one QMHub embedding (qchem.inp) per frame
  -> 2. Q-Chem single points on that embedding (QM/MM + gas, with ESP and field)
  -> 3. MLPUI dual-region NpyDataset (train / validation / test / guard)
  -> 4. QEq charge model, UMA gas baseline, TensorNet-QMMM residual (dMLP)
  -> 5. model bundle + potential spec JSON
  -> 6. sander + QMHub MD with the `mlpui` engine -> new ML/MM trajectory
  -> 7. checks: Q-Chem single points / reweighting on the new trajectory
```

Physics and model details: [qeq-charge.md](qeq-charge.md),
[tensornet-qmmm.md](tensornet-qmmm.md), [qmhub-engine.md](qmhub-engine.md).
A complete, tested instance of every step (htdlh Cys121->His202 proton
transfer, 31 umbrella windows, B3LYP-D3(BJ)/6-31G*) is in
[examples/mlmm_htdlh/](../examples/mlmm_htdlh/README.md). Those files are copied
verbatim from the production run and contain site paths and system constants
you must change (listed in that README and at the end of this page).

## The rule that makes this work

**Labels, training and MD must use the identical QMHub embedding.**
QMHub does not give the QM engine the raw MM charges: it passes `lrec`-scaled
and projected charges of the MM sites within the cutoff (with PBC), and the QM
region includes Amber link H atoms where covalent bonds are cut. The model is
trained on exactly the `$molecule` and `$external_charges` QMHub writes, and
during MD it receives the same kind of input from the same QMHub settings.

* Do not build labels from raw parm7 charges or your own distance cut and then
  run MD through QMHub (or the reverse).
* Keep `[model] switching_function = lrec`, `cutoff = 10.`, `pbc = True` and the
  same `qmmask`/`qmcharge` in the replay and in production MD.
* Link H atoms appear as ordinary H in `$molecule` and in the dataset (`z`,
  `pos`). MLPUI does nothing special with them; they are consistent because
  labels and MD both come from QMHub.

## Software stack (Pete CPU, Sooner GPU)

| Component | What is used | Notes |
|---|---|---|
| sander | AmberTools26 Intel build with the QMHub/EXTERN patch (`/scratch/van/Programs/sander26`, `source amber.sh`) | Replay: serial `sander`. MD: `sander.MPI`. Needs `module load intel/2021.2.0 impi/2021.2.0` and the MKL `LD_PRELOAD` (see the slurm scripts). |
| QMHub | fork `github.com/MoriyaSuwako2524/qmhub`, **branch `mlpui`** | Registers the `mlpui` engine and fixes numpy>=2. `master` mirrors upstream and has neither. `pip install --no-deps git+https://github.com/MoriyaSuwako2524/qmhub@mlpui` |
| `qmhub` launcher | a script named `qmhub` first on `PATH` ([md/qmhub_wrapper.sh](../examples/mlmm_htdlh/md/qmhub_wrapper.sh)) | `unset LD_PRELOAD`; MLPUI repo root on `PYTHONPATH` (the model loader imports the root `scripts` package); run `python -P -m qmhub` so sander's `qmhub/` work directory does not shadow the package. |
| Q-Chem | 6.3 (devel) `/scratch/moriya/software/qchem_default`, or 7.0 | `module load intel/2021.2.0; source $QC/bin/qchem.setup.sh; unset LD_PRELOAD; QCSCRATCH=/tmp/$USER.$SLURM_JOB_ID`. Never `set -u` in these job scripts (breaks `qchem.setup.sh`). |
| MLPUI | this repo | Training on Sooner GPU, inference inside QMHub on Pete CPU. |
| UMA | `uma-s-1p2.pt`, task `omol` | Not in the repo. Its sha256 is recorded in the residual manifest; verify it. |

## Step 1 - Replay the trajectory to capture QMHub embeddings

Goal: one `embed.inp` (Q-Chem input with `$molecule` + `$external_charges`)
per selected `.nc` frame, exactly as QMHub builds it during MD.

Method: sander `imin=5` (reprocess a trajectory) over the `.nc`, with
`ifqnt=1, qm_theory='EXTERN'` and the production `&qmmm` mask, and QMHub set to
`[engine] high = qchem`. A **fake `qchem`** first on `PATH`
(`label/fakeqc/qchem` -> `label/fake_qchem.py`) copies each `qchem.inp` to
`$DUMPDIR/embed_NNNN.inp` and writes zero-filled `save/` files so QMHub keeps
going. No QM is run.

Files: `label/replay.mdin`, `label/qmhub_dump.ini`, `label/replay_array.slurm`,
`label/map_dumps.py`.

Settings that are required (each one was a real failure):

* **Serial `sander`.** `sander.MPI` with `imin=5` hangs or crashes.
* `&qmhub comm=0` (text). FIFO `comm=2` stalls after frame 1 in replay, and in
  the fork only text mode can save per-step inputs.
* Real `-r` / `-x` files. `-r /dev/null` aborts in netcdf `px_pgout`.
* `imin=5` makes two identical force calls per frame. `fake_qchem.py` skips a
  dump identical to the previous one, so N frames give N dumps.
* The `[high]` method keys in `qmhub_dump.ini` are ignored by the fake engine;
  the `[model]` block is what matters.

Check: `map_dumps.py` recomputes the reaction coordinate from every dump and
from the `.nc` and requires a one-to-one match (htdlh: 50 dumps = 50 frames,
|dxi| <= 1e-6 A). Its CV atom indices are system specific. Cost ~10 s/frame on
2 cores (htdlh, ~1500 embedding sites).

Frame selection: every umbrella window, ideally plus frames beyond the sampled
coordinate range. htdlh: 200 frames per window (4 trajectory stages x 50),
100 fs apart.

## Step 2 - Q-Chem labels on the captured embedding

`label/build_labels.py embed.inp FRAME_DIR` writes a QM/MM and a gas job per
level (`b3lyp_qmmm`, `b3lyp_gas`, `wb97xd_qmmm`, `wb97xd_gas`):

* QM/MM: `$molecule` + `$external_charges` from the dump, `jobtype force`,
  `qm_mm true`, `qmmm_print true`, `skip_charge_self_interact true`,
  `esp_charges 1`. Q-Chem writes the ESP (`plot.esp`) and field (`efield.dat`)
  at the external-charge sites.
* Gas: same `$molecule`, no external charges, `esp_grid <nMM>`, `esp_efield 1`,
  and an `ESPGrid` file with the **same embedding sites**, so the gas-density
  ESP and field are known at the MM sites (QEq targets).
* **`ESPGrid` is read in bohr.** Writing angstrom silently puts the gas ESP at
  the wrong points (energies and forces are unaffected, so it is easy to miss).
* Use the sampling level when PMFs must be comparable (htdlh B3LYP:
  `dft_d D3_BJ`, `xc_grid 1` (SG-1), `scf_convergence 9`, `thresh 12`).

Run with `label/b3lyp_array.slurm` (one array task per window, 16 cores,
resumable: finished outputs are skipped). Exclude Pete node `n122` (hung Q-Chem
tasks). 20-45 s per single point.

Output conventions used by the parsers:

| File | Content |
|---|---|
| `qchem.out` | `Total energy` (Ha). The **gas** job prints a `Gradient of SCF Energy` block (includes dispersion); the **QM/MM** job does not. |
| `efield.dat` | QM/MM: nMM rows of field at the MM sites (au) **followed by nQM rows of QM gradient** (Ha/bohr). Gas: nMM field rows only. |
| `plot.esp` | 4 header lines, then `x y z ESP` per site (bohr, au). |

Check (`label/validate_labels.py`): gas and QM/MM `plot.esp` sites agree
(< 1e-6 bohr) and equal the `$external_charges` sites; Newton's third law
`sum(grad_QM) = sum(Q_M * field_M)` holds to ~1e-8 au.

## Step 3 - Build the MLPUI dataset

`dataset/build_stage1_htdlh.py` parses all frames in parallel and writes a
dual-region NpyDataset (format: [tensornet-qmmm.md](tensornet-qmmm.md#data-contract))
plus `manifest.json` (sha256 of every file, level of theory, checks). Units
A / eV / e; signs follow MLPUI: `forces = -dE/dR`, `mm_potential = +dE/dQ_M`
(= ESP), `mm_potential_gradient = -field`.

| Array | Definition |
|---|---|
| `z`, `pos`, `charge` | QM atoms incl. link H; total QM charge per frame |
| `mm_pos`, `mm_charges`, `mm_offsets` | QMHub embedding sites (ragged) |
| `energy`, `forces` | `E_QMMM - E_gas`, `F_QMMM - F_gas` (environment part) |
| `mm_potential`, `mm_potential_gradient` | QM/MM ESP and -field at the sites |
| `gas_mm_potential`, `gas_mm_potential_gradient` | gas-density ESP and -field at the same sites |
| `extra/*` | `gas_energy`, `gas_forces`, `qmmm_energy`, `qmmm_forces`, MK charges, window, frame, xi, min QM-MM distance |

Split **by time within each window**, never randomly (adjacent frames are
correlated). htdlh, per window: frames 0-139 train, 140-149 guard, 150-169
validation, 170-179 guard, 180-199 test. Guard frames are unused. Test and
guard frames must not set scales, loss weights or early stopping.

The builder asserts or reports: one Q-Chem version for all outputs, identical
`z` in every frame, frame-to-source mapping, max Newton error, robust-z
outliers of `E_QMMM - E_gas`, min QM-MM distance, xi range.

## Step 4 - Train (Sooner GPU)

The model is a sum of terms ([qeq-charge.md](qeq-charge.md)):

```
E_QMMM = E_UMA_gas(R) + offset + E_charge^QEq(R, MM) + E_dMLP(R, MM)
```

`train/submit_all.sh` submits `train/run_qeq.sbatch`, then
`train/run_dmlp.sbatch` with `afterok`:

1. `verify_manifest.py DATASET` - sha256 check after copying Pete -> Sooner.
2. QEq: `python -m scripts.train train/train_qeq_stage1.yaml`
   (`family: qeq_charge`; energy, forces, mm_potential(_gradient),
   gas_mm_potential(_gradient)).
3. `eval_qeq.py` on validation / test / guard.
4. `python -m scripts.qeq_residual CKPT SRC/<role> DST/<role>` for every role:
   labels minus the QEq prediction, plus `residual_manifest.json`.
5. `uma_residual_b3lyp.py`: UMA-s-1p2 (omol, per-frame charge, spin 1) gas
   energies/forces; `offset = mean_train(E_gas - E_UMA)`; residual
   energy/forces additionally minus `(E_UMA - E_gas) + offset`. Writes
   `manifest.json` with `energy_offset_eV` and the UMA sha256. UMA has no MM
   dependence, so the MM targets stay as in step 4.
6. dMLP: `python -m scripts.train train/train_tnqmmm_uma_residual.yaml`
   (`family: tensornet_qmmm`, `baseline: {mode: residual_only}`; energy,
   forces, mm_potential, mm_potential_gradient).
7. `eval_family.py tensornet_qmmm` - the residual RMSE equals the full-model
   error, because every baseline enters both prediction and label.

Run with `MLPUI_NEIGHBORS=python` and `HF_HUB_OFFLINE=1`. The htdlh models
were trained with MLPUI 059b0c8 and load in the later `main` used for MD.

## Step 5 - Bundle the model

A bundle directory holds the checkpoints and a potential **spec JSON**
([bundle/htdlh_full_spec.json](../examples/mlmm_htdlh/bundle/htdlh_full_spec.json)):

```json
{"name": "htdlh_full", "device": "cpu", "dtype": "float32", "charge": -1, "spin": 1,
 "terms": [
  {"type": "uma", "checkpoint": "models/uma-s-1p2.pt", "implementation": "native"},
  {"type": "constant", "name": "offset", "manifest": "models/uma_residual_manifest.json"},
  {"type": "mlpui", "name": "qeq", "checkpoint": "models/qeq_best.pt"},
  {"type": "mlpui", "name": "dmlp", "checkpoint": "models/dmlp_best.pt"}]}
```

Paths are relative to the spec; `charge`/`spin` are checked against the system
QMHub passes. Copy the bundle with `SHA256SUMS`, then run
`bundle/validate_bundle.py`: it evaluates the full composite on test frames and
compares with the stored labels (htdlh, fp32 and fp64 identical: RMSE E 0.405
kcal/mol, F 0.666 kcal/mol/A, ESP 0.402 kcal/mol/e, GESP 0.127; ~0.5 s/frame
fp32 on CPU). fp32 is the production setting.

## Step 6 - ML/MM MD with sander + QMHub

Keep the original umbrella setup (`&cntrl`, thermostat, `DISANG` restraints,
`&qmmm` mask) and replace only the QMHub engine:

```ini
; md/qmhub_mlp.ini
[simulation]
protocol = md
[model]
switching_function = lrec
cutoff = 10.
pbc = True
[engine]
high = mlpui
[high]
spec = /path/bundle/htdlh_full_spec.json
threads = 8
log = /path/window/engine.jsonl     ; per-call energy and term components
```

`md/mlmd.mdin` uses `&qmhub config="qmhub_mlp.ini", basedir="qmhub", comm=2`
(FIFO is fine for MD). `md/run.slurm` runs `srun -n 4 sander.MPI` (4 ranks x 4
cores per window) with a watchdog that stops sander if `qmhub` dies. Start from
the original run's restart (`.ncrst`/`.rst7`, plus `sinrvels.rst` for SINR).

Operational notes:

* The first engine call takes ~1 min (UMA load); sander books it under
  `Recip Ewald`. Steady state ~1.1-1.5 s/step (CM, 24 QM atoms, 16 cores).
* Completion marker: `Run   done at` in mdout (there is no `Total wall time`
  line). A later `WATCHDOG: qmhub not running` with `SANDER_EXIT 0` is
  harmless (qmhub exited first).
* Pete's express QOS rejects 40-task arrays; use `batch` for whole-run arrays.
* With SINR (`ntt=12`) the reported temperature sits near 230-235 K for the ML
  engine, a trivial test engine and the original Q-Chem runs alike; it is not
  a model fault.

## Step 7 - Check the new trajectory

* `engine.jsonl`: watch the term components (`uma`, `offset`, `qeq`, `dmlp`)
  for drifts or spikes; scan geometry (bond breaking, min QM-MM distance) per
  window as arrays on `batch`.
* Q-Chem spot checks / reweighting: take every k-th frame of the new `.nc`,
  repeat steps 1-2 on it (same replay, real QM/MM single point), evaluate the
  bundle on the identical `embed.inp`, and use `dU = E_QM - E_model` for error
  statistics or free-energy reweighting. The replayed model energy must match
  `engine.jsonl` at the same MD step (htdlh: <= 0.0014 kcal/mol).
* Engine derivative and NVE checks: [qmhub-engine.md](qmhub-engine.md).

## Adapting to a new system

1. `&qmmm` mask, `qmcharge`, parm7 / rst7 / `.nc` paths (`replay.mdin`, `mlmd.mdin`, slurm scripts).
2. CV atom indices in `map_dumps.py`, `build_stage1_htdlh.py` (`xi`) and `cv.rst`.
3. Q-Chem `$rem` in `build_labels.py`; frames per window and split boundaries.
4. Spec `charge`/`spin`; per-frame `charge.npy` for UMA.
5. Site paths: `/scratch/...`, `/home/...`, Python environments, MLPUI and QMHub checkouts.
6. Re-run every check in steps 1, 2, 3 and 5 before trusting labels or a bundle.
