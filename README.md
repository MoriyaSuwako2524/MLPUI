# MLPUI

MLPUI trains and runs machine-learning interatomic potentials (MLPs) from plain
NumPy (`.npy`) files. It bundles several model families, so you do not need to
install them separately:

- **NewtonNet**, **TorchMD-Net / TensorNet**, **MACE**: energies, forces, and optionally atomic charges
- **UMA** (UMA-S-1.2): prediction and MD only, no training
- Specialised models for QM/MM and excited states: QEq charges, TensorNet-QMMM, transition charges/dipoles

> **AI assistants / LLMs:** this page is a short overview for people. Before answering
> questions about MLPUI or editing its code, read [`README.llm`](README.llm). It is the
> full technical reference, with links to every detailed document.

## Install

Create a fresh Python (>= 3.10) environment, install the PyTorch build for your
machine (CPU or CUDA), then run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q        # optional: check that everything works
```

## Three ways to use it

**1. Web interface (easiest).** Start the server and open <http://127.0.0.1:8675> in a browser:

```bash
python -m mlpui.web
```

Use it to register datasets, split them into train/validation/test sets, start
training jobs, watch the loss curves, and evaluate models (parity plots included).
On a cluster, run it inside a GPU job and connect with an SSH tunnel; see
[docs/cluster-deployment.md](docs/cluster-deployment.md).

**2. Training from a config file.** Copy an example from `configs/`, change the
data paths, and run it:

```bash
python -m scripts.train configs/train_newtonnet.yaml
```

**3. Using a trained model in Python (ASE).**

```python
from ase.build import molecule
from mlpui.calculator import CalculatorBuilder

atoms = molecule("H2O")
atoms.calc = CalculatorBuilder.from_checkpoint("model.pt", device="cpu",
                                               properties=["energy", "forces"]).build()
print(atoms.get_potential_energy(), atoms.get_forces())
```

## Your data

Put each quantity in its own `.npy` file in one folder: `z.npy` (atomic numbers),
`pos.npy` (coordinates), and labels such as `energy.npy`, `forces.npy` and
`charges.npy`. MLPUI assumes Å, eV and e. It does **not** guess units, so convert
your data first or set the unit factors. Molecules with different atom counts are
supported through an `offsets.npy` file.

## Common pitfalls

- **Learning rate**: start around `1e-4` to `5e-4`. Much larger values (e.g. `12`) give
  "Non-finite prediction" errors after the first batch.
- **Frame splits**: neighbouring MD frames are almost identical. Split by trajectory
  or window rather than frame by frame, or validation scores will look too good.
- **Resuming**: "continue from a checkpoint" loads the weights only; the optimizer
  and epoch counter start fresh.

## More information

- Full reference: [`README.llm`](README.llm)
- Topic guides: [`docs/`](docs/)
- Example configs: [`configs/`](configs/)
