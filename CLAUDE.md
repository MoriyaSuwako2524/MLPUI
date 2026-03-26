# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MLPUI** (Machine Learning Potentials User Interface) is a PyTorch-based framework for inference and fine-tuning of machine learning potential (MLP) models, with ASE (Atomic Simulation Environment) calculator integration. Primary use case: computing energies and forces for molecular dynamics.

## Installation & Setup

```bash
pip install -e .
```

Key dependencies: `torch>=2.0`, `pytorch-lightning>=2.0`, `torch_geometric>=2.0`, `ase`, `numpy`, `pyyaml`, `omegaconf`, `psutil`.

## Running Notebooks

Usage examples live in `notebooks/`:
```bash
jupyter notebook notebooks/calc_test.ipynb
```

There is no formal test suite — notebooks serve as integration tests.

## Architecture

### Core Data Flow

```
Checkpoint (.pt) → load_torch_file() → detect_unet_config() → model_config_from_unet()
→ BaseModel/UMA → ModelPatcher → CalculatorBuilder → MLPCalculator (ASE-compatible)
```

### Key Modules

**`mlpui/calculator.py`** — Main entry point for users.
- `CalculatorBuilder.from_checkpoint(ckpt_path, task, ...)` — factory that auto-detects model family and wires up the right input/output adapters
- `InputAdapter` / `OutputAdapter` — abstract interfaces bridging ASE `Atoms` objects ↔ model tensors. `UMAInputAdapter` and `SimpleInputAdapter` are the two implementations.
- `UMACalculator` — specialized calculator for UMA/eSCNMD models

**`mlpui/model_patcher.py`** — Non-destructive weight patching for fine-tuning.
- `ModelPatcher.clone()` creates lightweight copies sharing a backbone but with independent patch sets — used for multi-property prediction from one foundation model
- Patch types: `FULL` (replacement), `DELTA` (additive: `W += strength * delta`), `LORA` (not yet implemented)

**`mlpui/model_loader.py`** — Safe checkpoint loading with a whitelist of allowed unpickling globals. Handles PyTorch Lightning checkpoint formats (`.ema_state_dict`, `.model_state_dict` keys).

**`mlpui/model_detection.py`** — Auto-detects model architecture config from state dict keys. Outputs feed into `supported_models.py` to select the correct model class.

**`mlpui/model_management.py`** — Device and VRAM management (`VRAMState`, `CPUState` enums). `get_torch_device()` and `unet_dtype()` handle automatic device/precision selection.

**`mlpui/models/uma/`** — The primary model family: `eSCNMDBackbone` (equivariant graph neural network with SO(3) symmetry). Config keys: `num_layers`, `sphere_channels`, `hidden_channels`, `cutoff`, `max_num_elements`.

### Configuration System

YAML configs live in `configs/`. The config loader reads from `./configs/` or `$CHEMLAB_CONFIG_DIR`. Key files:
- `model_loader.yaml` — mmap settings, safe unpickling globals whitelist
- `model_management.yaml` — device/CPU-only overrides

### Adapter Pattern

When adding support for a new model family, implement both:
1. `InputAdapter.prepare_input(atoms, **kwargs) -> dict` — ASE Atoms → model input tensors
2. `OutputAdapter.parse_output(output, atoms) -> dict` — model output → ASE results dict (`energy`, `forces`, `stress`)

Then register in `supported_models.py` and add detection logic in `model_detection.py`.

## Known Issues

- `model_detection.py` line ~95 has a syntax/logic issue (missing body after an `if` statement before `datasets = [...]`)
- Several `print()` debug statements remain in `calculator.py` that should eventually be replaced with `logging`
