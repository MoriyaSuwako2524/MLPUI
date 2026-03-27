"""
Dataset manager for NPZ-format molecular datasets.

Supports standard fields:
  positions / R / coords       – (N_frames, N_atoms, 3)  Cartesian coords [Å]
  numbers   / Z / z            – (N_atoms,) or (N_frames, N_atoms)  atomic numbers
  energy    / E / energies     – (N_frames,)  total energy [eV]
  forces    / F / force        – (N_frames, N_atoms, 3)  forces [eV/Å]
  charges   / Q / partial_charges – (N_frames,) or (N_frames, N_atoms)  charges [e]
  dipole    / D / dipole_moment – (N_frames, 3)  dipole vector [e·Å]
  cell      / lattice          – (N_frames, 3, 3) or (3, 3)  lattice vectors [Å]
  pbc                          – (N_frames, 3) or (3,)  periodic boundary flags
  stress    / virial           – (N_frames, 3, 3) or (N_frames, 6)  stress tensor
"""
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Key alias table ───────────────────────────────────────────────────────────

_ALIASES: Dict[str, List[str]] = {
    "positions": ["positions", "R", "coords", "coordinates", "pos"],
    "numbers":   ["numbers", "Z", "z", "atomic_numbers", "species", "atoms","qm_type"],
    "energy":    ["energy", "E", "energies", "total_energy", "Energy"],
    "forces":    ["forces", "F", "force", "Forces","G","gradient","gradients","Gradients","Gradient"],
    "charges":   ["charges", "Q", "partial_charges", "charge", "mulliken"],
    "dipole":    ["dipole", "D", "dipole_moment", "dipoles", "mu"],
    "cell":      ["cell", "lattice", "cells", "lattice_vectors"],
    "pbc":       ["pbc", "periodic", "PBC"],
    "stress":    ["stress", "virial", "Stress", "stresses"],
}

_ELEMENT_SYMBOLS = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",
    9: "F", 10: "Ne",11: "Na",12: "Mg",13: "Al",14: "Si",15: "P", 16: "S",
    17:"Cl",18: "Ar",19: "K", 20: "Ca",21: "Sc",22: "Ti",23: "V", 24: "Cr",
    25:"Mn",26: "Fe",27: "Co",28: "Ni",29: "Cu",30: "Zn",31: "Ga",32: "Ge",
    33:"As",34: "Se",35: "Br",36: "Kr",37: "Rb",38: "Sr",39: "Y", 40: "Zr",
    47:"Ag",50: "Sn",53: "I", 78: "Pt",79: "Au",80: "Hg",82: "Pb",83: "Bi",
}


def _resolve(npz_keys: List[str], canonical: str) -> Optional[str]:
    """Return the first matching alias found in npz_keys, or None."""
    for alias in _ALIASES.get(canonical, [canonical]):
        if alias in npz_keys:
            return alias
    return None


def _numbers_to_formula(numbers) -> str:
    counts = Counter(int(n) for n in numbers)
    # Hill order: C first, H second, then alphabetical
    parts: List[str] = []
    for z in sorted(counts, key=lambda z: (z not in (6, 1), _ELEMENT_SYMBOLS.get(z, f"Z{z}"))):
        sym = _ELEMENT_SYMBOLS.get(z, f"Z{z}")
        c = counts[z]
        parts.append(f"{sym}{c}" if c > 1 else sym)
    return "".join(parts)


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None so the result is valid JSON."""
    if isinstance(obj, float):
        import math
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def _to_json(val) -> Any:
    """Convert numpy scalar/array to a JSON-serialisable Python type (NaN → null)."""
    if isinstance(val, np.ndarray):
        return _sanitize(val.tolist())
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        import math
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(val, np.bool_):
        return bool(val)
    return val


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FieldInfo:
    shape: List[int]
    dtype: str


@dataclass
class DatasetInfo:
    name: str
    num_frames: int
    fields: Dict[str, FieldInfo]   # raw NPZ key → FieldInfo
    canonical_map: Dict[str, str]  # canonical name → actual NPZ key


# ── Manager ───────────────────────────────────────────────────────────────────

class DatasetManager:
    """
    Manages a directory of NPZ molecular-dynamics datasets.

    Parameters
    ----------
    dataset_dir : str
        Path to the folder that holds *.npz files.
    """

    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)

    # ── Listing & info ────────────────────────────────────────────────────────

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return a list of summary dicts for every .npz file."""
        results = []
        if not os.path.isdir(self.dataset_dir):
            return results
        for fname in sorted(os.listdir(self.dataset_dir)):
            if not fname.lower().endswith(".npz"):
                continue
            name = fname[:-4]
            try:
                info = self._load_info(name)
                results.append({
                    "name":       name,
                    "num_frames": info.num_frames,
                    "fields":     list(info.fields.keys()),
                    "canonical":  list(info.canonical_map.keys()),
                })
            except Exception as exc:
                results.append({"name": name, "error": str(exc)})
        return results

    def get_info(self, name: str) -> Dict[str, Any]:
        """Return detailed info dict for a named dataset."""
        info = self._load_info(name)
        return {
            "name":       info.name,
            "num_frames": info.num_frames,
            "fields": {
                k: {"shape": v.shape, "dtype": v.dtype}
                for k, v in info.fields.items()
            },
            "canonical": info.canonical_map,
        }

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_frames(self, name: str, start: int = 0, count: int = 50) -> List[Dict[str, Any]]:
        """Return a list of summary dicts, one per frame in [start, start+count)."""
        path = self._get_path(name)
        frames: List[Dict[str, Any]] = []

        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.keys())
            cm   = self._build_canonical_map(keys)
            n    = self._count_frames(npz, keys, cm)
            end  = min(start + count, n)

            pos_k    = cm.get("positions")
            num_k    = cm.get("numbers")
            energy_k = cm.get("energy")
            forces_k = cm.get("forces")
            dipole_k = cm.get("dipole")
            charge_k = cm.get("charges")

            for i in range(start, end):
                row: Dict[str, Any] = {"index": i}

                # Formula & atom count
                if num_k is not None:
                    nums_arr = npz[num_k]
                    nums = nums_arr[i] if nums_arr.ndim == 2 else nums_arr
                    row["formula"]   = _numbers_to_formula(nums)
                    row["num_atoms"] = int(len(nums))

                # Energy
                if energy_k is not None:
                    v = float(npz[energy_k][i])
                    import math
                    row["energy"] = None if math.isnan(v) or math.isinf(v) else v

                # Max force magnitude — handle both 3D (N_atoms,3) and 4D (M,N_atoms,3)
                if forces_k is not None:
                    f = npz[forces_k][i]        # (N_atoms,3) or (M,N_atoms,3)
                    if f.ndim == 3:
                        # Use first slice that has finite values, or mean over slices
                        f = f[0]                # (N_atoms, 3) - first gradient
                    norms = np.linalg.norm(f, axis=-1)
                    finite = norms[np.isfinite(norms)]
                    row["max_force"] = float(np.max(finite)) if len(finite) else None

                # Dipole magnitude
                if dipole_k is not None:
                    d = npz[dipole_k][i]
                    norm_val = float(np.linalg.norm(d))
                    import math
                    row["dipole_norm"] = None if math.isnan(norm_val) else norm_val

                # Total charge (sum of partial charges, or scalar)
                if charge_k is not None:
                    q_arr = npz[charge_k]
                    if q_arr.ndim == 2:
                        row["total_charge"] = float(np.sum(q_arr[i]))
                    else:
                        row["total_charge"] = float(q_arr[i])

                frames.append(row)

        return frames

    def get_frame(self, name: str, index: int) -> Dict[str, Any]:
        """Return all field data for a single frame."""
        path = self._get_path(name)

        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.keys())
            cm   = self._build_canonical_map(keys)
            n    = self._count_frames(npz, keys, cm)

            if not (0 <= index < n):
                raise IndexError(f"Frame index {index} out of range [0, {n})")

            result: Dict[str, Any] = {"index": index}

            for canonical, npz_key in cm.items():
                arr = npz[npz_key]
                if arr.ndim == 0:
                    result[canonical] = _to_json(arr)
                elif arr.ndim == 1:
                    # Could be per-frame scalar OR shared 1-D array (e.g. numbers)
                    if len(arr) == n and canonical not in ("numbers",):
                        result[canonical] = _to_json(arr[index])
                    else:
                        result[canonical] = _to_json(arr)
                else:
                    # First axis is frame axis
                    result[canonical] = _to_json(arr[index])

            # Also add raw keys not covered by canonical map
            for k in keys:
                if k not in cm.values():
                    arr = npz[k]
                    if arr.ndim >= 1 and arr.shape[0] == n:
                        result[k] = _to_json(arr[index])
                    else:
                        result[k] = _to_json(arr)

        return result

    # ── Mutation ──────────────────────────────────────────────────────────────

    def upload(self, filename: str, content: bytes) -> str:
        """Save uploaded bytes as a .npz file; returns the stored filename."""
        safe = os.path.basename(filename)
        if not safe.lower().endswith(".npz"):
            raise ValueError("Only .npz files are accepted")
        dest = os.path.join(self.dataset_dir, safe)
        with open(dest, "wb") as f:
            f.write(content)
        return safe[:-4]  # return name without extension

    def delete(self, name: str) -> None:
        """Delete a dataset by name."""
        path = self._get_path(name)
        os.remove(path)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_path(self, name: str) -> str:
        safe = os.path.basename(name)
        path = os.path.join(self.dataset_dir, safe + ".npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset '{name}' not found")
        return path

    def _load_info(self, name: str) -> DatasetInfo:
        path = self._get_path(name)
        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.keys())
            cm   = self._build_canonical_map(keys)
            n    = self._count_frames(npz, keys, cm)
            fields = {
                k: FieldInfo(shape=list(npz[k].shape), dtype=str(npz[k].dtype))
                for k in keys
            }
        return DatasetInfo(name=name, num_frames=n, fields=fields, canonical_map=cm)

    def _build_canonical_map(self, keys: List[str]) -> Dict[str, str]:
        cm: Dict[str, str] = {}
        for canonical in _ALIASES:
            hit = _resolve(keys, canonical)
            if hit is not None:
                cm[canonical] = hit
        return cm

    def _count_frames(self, npz, keys: List[str], cm: Dict[str, str]) -> int:
        # Prefer energy (1-D), then positions (3-D first axis)
        for canon in ("energy", "forces", "positions"):
            if canon in cm:
                arr = npz[cm[canon]]
                return arr.shape[0]
        # Fallback: first key whose first axis is > 1
        for k in keys:
            if npz[k].ndim >= 1:
                return npz[k].shape[0]
        return 0
