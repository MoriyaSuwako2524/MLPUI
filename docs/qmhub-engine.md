# Composite QM/MM potentials and the QMHub `mlpui` engine

`mlpui.potentials` turns trained MLPUI checkpoints (plus UMA and simple terms)
into one QM/MM potential, and `mlpui/potentials/qmhub.py` exposes it as a QMHub
QM engine so sander can run ML/MM MD. This is the inference side of
[mlmm-workflow.md](mlmm-workflow.md) (steps 5-6). Note: the "no MD adapter"
statements in [tensornet-qmmm.md](tensornet-qmmm.md) refer to the ASE
calculator path of the backend; MD runs through this module instead.

## Term contract (`mlpui/potentials/base.py`)

A `PotentialTerm` maps one `QMMMInput(numbers, positions, charge, spin,
mm_positions, mm_charges)` to additive properties in A / eV / e:

| Property | Shape | Definition |
|---|---|---|
| `energy` | scalar | eV |
| `forces` | [N,3] | -dE/dR_QM |
| `mm_potential` | [M] | +dE/dQ_M (ESP, eV/e) |
| `mm_potential_gradient` | [M,3] | GESP: gradient of the fixed (already polarized) QM response potential at r_M = -field (Q-Chem `ESP_EFIELD` convention) |
| `mm_forces` | [M,3] | -dE/dr_M |

`CompositePotential` sums terms; per-term energies are kept in `components`.
`MMDependence` declares how a term depends on MM sites: `NONE`,
`LINEAR_CHARGE` (E depends on site M only through Q_M * K(r_M), so
dE/dr_M = Q_M * GESP_M exactly) or `GENERAL`. **QMHub's MM-gradient chain rule
assumes `LINEAR_CHARGE` or `NONE`**; the engine refuses `GENERAL` terms. QEq and
TensorNet-QMMM are `LINEAR_CHARGE` (their MM input enters only through
charge-weighted kernels).

## Spec JSON (`mlpui/potentials/spec.py`)

```json
{"name": "cm_full", "device": "cpu", "dtype": "float32", "charge": -2, "spin": 1,
 "terms": [
   {"type": "uma", "checkpoint": "models/uma-s-1p2.pt", "implementation": "native"},
   {"type": "constant", "name": "offset", "manifest": "models/uma_residual_manifest.json"},
   {"type": "mlpui", "name": "qeq", "checkpoint": "models/best.pt"},
   {"type": "mlpui", "name": "dmlp", "checkpoint": "models/dmlp_best.pt"}]}
```

* Paths are relative to the spec file. Top-level `device` / `dtype` / `charge`
  / `spin` are defaults for terms; `charge`/`spin` are also checked by the
  QMHub engine against the system it receives.
* `uma`: gas-phase UMA (`task` default `omol`; `implementation` `native`
  (bundled predictor) or `fairchem`). Needs the total QM charge.
* `constant`: `energy_eV`, or `manifest` + key `energy_offset_eV` (the
  train-mean `E_ref_gas - E_UMA` offset written by the residual step).
* `mlpui`: a `qeq_charge` or `tensornet_qmmm` checkpoint trained with
  `baseline: {mode: residual_only}` (other families/modes are rejected).
* `fixed_charge_coulomb`: fixed QM point charges (`qm_charges`); a simple
  baseline and an analytic test oracle.
* `composite`: nested `terms`. `python`: `{"target": "pkg.module:factory",
  "kwargs": {...}}`, the factory must return a `PotentialTerm`.

Python use (offline evaluation, bundle validation):

```python
from mlpui.potentials import QMMMInput, build_potential, load_spec
pot = build_potential(load_spec("bundle/spec.json"))
out = pot.evaluate(QMMMInput(z, pos, -1, 1, mm_pos, mm_q),
                   ["energy", "forces", "mm_potential", "mm_potential_gradient"])
```

## QMHub engine

With the QMHub fork (`github.com/MoriyaSuwako2524/qmhub`, branch `mlpui`) the
engine is registered natively; use a normal `qmhub.ini`:

```ini
[simulation]
protocol = md
[model]
switching_function = lrec
cutoff = 10.
pbc = True
[engine]
high = mlpui
[high]
spec = /path/model_spec.json
device = cpu                 ; optional override of the spec default
dtype = float32              ; optional override
threads = 8                  ; torch threads (default: QMHub nproc)
log = /path/engine.jsonl     ; optional: per-call time, n_mm, energy, term components
dump = /path/dump_dir        ; optional: per-call npz of inputs and outputs (large)
```

With upstream panxl/qmhub (numpy<2 only), put a script named `qmhub` first on
`PATH` that runs `python -m mlpui.potentials.qmhub "$@"`
(e.g. `python -m mlpui.potentials.qmhub qmhub.ini --fifo qmmm.inp --driver sander`).
Upstream QMHub 06fdbe7 fails to import under numpy>=2; the fork fixes this.

Units returned to QMHub: energy Ha, QM gradient Ha/bohr, ESP Ha/e, ESP
gradient Ha/(e bohr) at the embedding sites (same as QMHub's `qchem` and
`torch` engines). The model is loaded once per process; in FIFO mode
(`comm=2`) it stays resident for the whole run.

Launcher requirements on Pete (see
[examples/mlmm_htdlh/md/qmhub_wrapper.sh](../examples/mlmm_htdlh/md/qmhub_wrapper.sh)):
`unset LD_PRELOAD` (sander's MKL preload breaks torch), MLPUI repo root on
`PYTHONPATH`, and `python -P -m qmhub` so that sander's work directory named
`qmhub/` is not imported in place of the package.

## Verification done on this engine

* `tests/test_potentials.py`, `tests/test_potentials_qmhub.py`: QMHub end to
  end; total-energy finite differences match the QMHub gradient for QM and
  near-field MM atoms, with PBC on and off.
* Pete, CM window 20 (24 QM + 49156 MM atoms), finite differences through
  `qmhub --text` on sander's real `qmmm.inp` (fp64, h = 1e-3 A): QM atoms
  max deviation <= 1.2e-6 Ha/bohr (relative 4e-5); MM atoms at 1.5, 9.5 and
  12 A relative <= 7e-7.
* NVE, 50 steps, dt 1 fs: the model and a trivial fixed-charge engine give the
  same total-energy drift, so the drift is from the MM setup, not the engine.
* Timing: first call ~1 min (UMA load + QMHub setup), then ~0.6-0.7 s per
  engine call on 8 threads (CM); ~1.0-1.5 s per MD step overall.
