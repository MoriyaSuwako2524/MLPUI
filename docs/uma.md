# Native UMA molecular inference and NVT

MLPUI loads the complete UMA backbone and dataset-specific energy head internally.
No `fairchem` package, web server, remote service, model download, or API call is
required at runtime. PyTorch, ASE and the project's numerical dependencies are
still required. The supported validation target is `uma-s-1p2.pt`, task `omol`,
nonperiodic molecules, with energy (eV) and forces (eV/Angstrom). Other tasks,
periodic systems, stress, and training are not supported by this new interface.

## Setup

Use Python 3.12 and CUDA-enabled PyTorch appropriate for your GPU. On the local
Windows validation machine, an isolated environment was created in `.venv` with
PyTorch 2.11.0+cu128, ASE 3.29.0, and the declared MLPUI dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The environment contains no FAIRChem installation. A separate environment is
used only for optional upstream comparison during development.

## Calculator

```python
from ase.io import read
from mlpui.calculator import CalculatorBuilder

atoms = read('molecule.xyz')
atoms.calc = CalculatorBuilder.from_checkpoint(
    'C:/Users/suwak/Documents/uma/uma-s-1p2.pt',
    task='omol', charge=0, spin=1, device='cuda',
).build()
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

Charge is the integer total molecular charge. Spin is the multiplicity `2S+1`,
not the number of unpaired electrons; the molecular default is 1. These are
calculator settings; create another calculator to change them. Inference uses
EMA weights with strict key/shape matching, including all prediction heads.
Unknown checkpoint architectures and missing weights fail rather than silently
using random parameters. `dtype=torch.float64` is available for numerical checks.

The energy normalizer and fitted elemental offsets come from `tasks_config`
inside the checkpoint. The supplied `form_elem_refs.yaml` is **not** added to
molecular total energies. Single atoms are a special case: pass
`atom_refs='.../iso_atom_elem_refs.yaml'` to use tabulated DFT atomic energies
and zero forces, as in FAIRChem. A missing element/charge reference is an error.
For this single-atom convention, spin is ignored with an explicit warning.

## NVT command

From the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.md.nvt --xyz molecule.xyz --checkpoint C:/Users/suwak/Documents/uma/uma-s-1p2.pt --temperature-k 300 --duration-ps 1 --timestep-fs 0.5 --friction-per-ps 10 --charge 0 --spin 1 --device cuda --output runs/molecule_300K
```

Replace `--duration-ps 1` with `--steps 2000` to request steps. The duration must
be an integer multiple of the timestep. `--check-only` checks energy and forces
without integrating. `--save-every`, `--log-every`, `--seed`, `--frame`, and
`--fix-com` are configurable; `--help` documents all options. The generic runner
has `--periodic`, but this native UMA interface deliberately rejects it.

The runner uses ASE Langevin at fixed geometry boundary conditions. Coordinates
are in Angstrom; simulation time is in ps and timestep in fs. The default
friction, 10/ps, corresponds to a 100 fs damping time. Velocities are freshly
sampled; this is not a restart runner. There is no automatic minimization or
equilibration. The default thermostat includes all Cartesian degrees of freedom;
`--fix-com` adds an explicit ASE FixCom constraint.

The output directory must be new. It contains `trajectory.traj`, `thermo.csv`,
`final.extxyz`, and `run.json`. The initial and successful final frames are saved
even if the requested output interval does not divide the number of steps.
Failed or interrupted runs retain previous outputs and record their status.

## Validation

```powershell
$env:MLPUI_UMA_CHECKPOINT = 'C:/Users/suwak/Documents/uma/uma-s-1p2.pt'
.\.venv\Scripts\python.exe -m pytest tests/test_native_uma.py -q -s
```

The real-model tests check all water force components by central finite
differences, translation invariance, net force, strict head loading, invalid
inputs, isolated atom handling, and an actual 20-step Langevin trajectory.
Without the environment variable, real-model tests are skipped.

Local GPU validation on 2026-09-29:

| Check | Result |
|---|---|
| Native vs FAIRChem, six geometries | Maximum energy difference 3.58e-7 eV; maximum force-component difference 6.56e-7 eV/Angstrom |
| Central finite differences, float64, displacement 1e-4 Angstrom | Maximum force error 2.25e-7 eV/Angstrom |
| Molecular dynamics | 100 water steps at 300 K target, 0.1 fs timestep, all finite; trajectory and log saved |
| Native UMA tests with real checkpoint | 5 passed |
| Project regression suite | 111 passed, 20 skipped (optional backend/integration conditions) |

Both numerical comparison processes used CUDA PyTorch 2.11.0 and quaternion
Wigner rotations; the reference package was FAIRChem 2.17.0. Its package metadata
pins PyTorch 2.8, so this is an observed numerical comparison with 2.11, not a
claim of upstream support for that combination. MLPUI used e3nn 0.4.4 and the
separate reference environment used e3nn 0.6.0. The short MD run verifies the
execution path and finite outputs, not equilibration or long-time stability.

Local artifacts are under `runs/uma-validation/`: `native.json`, `fairchem.json`,
`comparison.json`, `water.xyz`, and `water-nvt/`. These local files and model
weights are not tracked in Git.

`scripts/check_uma_reference.py` exports predictions on water, distorted and
rotated water, methane, ammonia and hydroxide. Run it with `--backend native`
in MLPUI and `--backend fairchem` in a separate FAIRChem environment, using the
same checkpoint. FAIRChem is imported only in the explicit reference mode.

## Implementation and attribution

The head architecture, dataset-to-expert order, energy normalization and
gradient-force equations follow FAIRChem 2.17.0, revision
`be54a56` (https://github.com/facebookresearch/fairchem/tree/fairchem_core-2.17.0).
The head's single-system one-hot routing is evaluated as direct expert selection.
All experts remain present for strict checkpoint loading. The existing MLPUI UMA
backbone is retained; compilation, expert merging and TF32 are not enabled by
the predictor. GPU results may have small floating-point variation.

Upstream copyright and MIT license are retained in `mlpui/models/uma/LICENSE`.
This covers source code; model weights retain their own upstream terms and are
not distributed with MLPUI.
