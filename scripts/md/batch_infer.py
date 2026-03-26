"""Batch inference from numpy structure files or arrays.

Usage
-----
    from scripts.md.batch_infer import run_batch

    # Minimal — files or arrays, charge/spin default to 0
    results = run_batch("positions.npy", "numbers.npy", calc)
    results = run_batch(pos_array, numbers_array, calc)

    # Per-structure charge and spin
    results = run_batch(pos_array, numbers_array, calc,
                        charge=charge_array,   # int or (B,) int array
                        spin=spin_array)       # int or (B,) int array

    # results is a list of dicts, one per structure:
    # [{"energy": float, "forces": ndarray(N,3)}, ...]

Charge and spin are stored in ``atoms.info`` and read by the calculator
at inference time, so they travel with the structure rather than being
fixed in the calculator.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ase import Atoms
from tqdm import tqdm


def _load_array(src: str | np.ndarray) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return src
    return np.load(src)


def _broadcast_int_param(value: int | np.ndarray | None, batch_size: int, name: str) -> np.ndarray:
    """Return an integer array of shape (B,) from a scalar, array, or None."""
    if value is None:
        return np.zeros(batch_size, dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        return np.full(batch_size, int(arr), dtype=np.int64)
    if arr.shape != (batch_size,):
        raise ValueError(
            f"'{name}' must be a scalar or 1-D array of length {batch_size}, "
            f"got shape {arr.shape}."
        )
    return arr


def load_structures(
    pos: str | np.ndarray,
    numbers: str | np.ndarray,
    charge: int | np.ndarray | None = None,
    spin: int | np.ndarray | None = None,
) -> list[Atoms]:
    """Build a list of ASE Atoms from positions and atomic numbers.

    Charge and spin are embedded in ``atoms.info`` so the calculator can
    read them at inference time without being reconfigured.

    Parameters
    ----------
    pos:
        ``(N, 3)`` or ``(B, N, 3)`` array, or path to a .npy file.
    numbers:
        ``(N,)`` or ``(B, N)`` array, or path to a .npy file.
        A 1-D array is broadcast to all structures in the batch.
    charge:
        Total charge per structure. Scalar (shared) or ``(B,)`` int array.
        Defaults to 0.
    spin:
        Total spin per structure. Scalar (shared) or ``(B,)`` int array.
        Defaults to 0.

    Returns
    -------
    list of :class:`ase.Atoms`, each with ``info["charge"]`` and
    ``info["spin"]`` set.
    """
    pos_arr = _load_array(pos).astype(np.float64)
    num_arr = _load_array(numbers)

    if pos_arr.ndim == 2:
        pos_arr = pos_arr[np.newaxis]
    if num_arr.ndim == 1:
        num_arr = np.tile(num_arr, (len(pos_arr), 1))

    B = pos_arr.shape[0]

    if pos_arr.shape[0] != num_arr.shape[0]:
        raise ValueError(
            f"Batch size mismatch: pos has {pos_arr.shape[0]} structures "
            f"but numbers has {num_arr.shape[0]}."
        )
    if pos_arr.shape[1] != num_arr.shape[1]:
        raise ValueError(
            f"Atom count mismatch: pos has {pos_arr.shape[1]} atoms "
            f"but numbers has {num_arr.shape[1]}."
        )

    charges = _broadcast_int_param(charge, B, "charge")
    spins   = _broadcast_int_param(spin,   B, "spin")

    structures = []
    for p, z, q, s in zip(pos_arr, num_arr, charges, spins):
        atoms = Atoms(numbers=z.astype(int), positions=p)
        atoms.info["charge"] = int(q)
        atoms.info["spin"]   = int(s)
        structures.append(atoms)
    return structures


def run_batch(
    pos: str | np.ndarray,
    numbers: str | np.ndarray,
    calc,
    properties: Sequence[str] = ("energy", "forces"),
    charge: int | np.ndarray | None = None,
    spin: int | np.ndarray | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """Run inference on a batch of structures.

    Parameters
    ----------
    pos:
        ``(N, 3)`` or ``(B, N, 3)`` positions array, or path to a .npy file.
    numbers:
        ``(N,)`` or ``(B, N)`` atomic numbers array, or path to a .npy file.
        A 1-D array is shared across all structures in the batch.
    calc:
        A :class:`~mlpui.calculator.MLPCalculator` instance.
    properties:
        Properties to compute (default: energy and forces).
    charge:
        Total charge per structure. Scalar (shared) or ``(B,)`` int array.
        Defaults to 0 for all structures.
    spin:
        Total spin per structure. Scalar (shared) or ``(B,)`` int array.
        Defaults to 0 for all structures.
    show_progress:
        Whether to show a tqdm progress bar (default: True).

    Returns
    -------
    list of dicts, one per structure, containing the computed properties.
    """
    structures = load_structures(pos, numbers, charge=charge, spin=spin)
    properties = list(properties)
    results = []

    for atoms in tqdm(structures, desc="Inferring", disable=not show_progress):
        atoms.calc = calc
        atoms.get_properties(properties)
        results.append(dict(atoms.calc.results))

    return results
