"""Numeric .npy datasets shared by all training backends (no pickle)."""
from pathlib import Path
import json

import numpy as np
from ase import Atoms
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    """A directory of z/pos/target arrays, optionally cell/pbc/charge/spin.

    Dense: pos[S,N,3], z[N] or z[S,N]. Ragged: pos[A,3], z[A],
    offsets[S+1]. Atomic targets follow the positions layout. All samples
    use the same target fields; units must already match the model.
    """

    targets = ("energy", "forces", "charges", "dipole", "stress", "mm_forces", "mm_potential", "mm_potential_gradient")
    mm_fields = ("mm_pos", "mm_charges", "mm_offsets")
    baseline_fields = ("baseline_energy", "baseline_forces", "baseline_mm_forces", "baseline_mm_potential", "baseline_mm_potential_gradient")

    def __init__(self, directory, *, files=None, gradients=False, energy_scale=1.0,
                 length_scale=1.0, baseline_metadata=None):
        self.directory = Path(directory)
        self.gradients = gradients
        self.energy_scale, self.length_scale = energy_scale, length_scale
        if not all(np.isfinite(v) and v > 0 for v in (energy_scale, length_scale)):
            raise ValueError("Unit scales must be finite and positive")
        names = ("z", "pos", "offsets", "cell", "pbc", "charge", "spin") + self.targets + self.mm_fields + self.baseline_fields + ("frame_index", "source_index", "short_contact")
        if files is not None and set(files) - set(names):
            raise ValueError(f"Unknown dataset fields: {set(files) - set(names)}")
        self.arrays = {}
        for name in names:
            if files is not None and name not in files:
                continue
            path = self.directory / (files[name] if files is not None else f"{name}.npy")
            if path.suffix != ".npy":
                raise ValueError("All dataset files must be .npy files")
            if files is not None and not path.is_file():
                raise FileNotFoundError(path)
            if path.exists():
                value = np.load(path, allow_pickle=False, mmap_mode="r")
                if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                    raise ValueError(f"{name}.npy must contain finite real numeric values")
                self.arrays[name] = value
        self.baseline_metadata = None
        if set(self.baseline_fields) & self.arrays.keys():
            metadata_path = self.directory / "baseline_metadata.json"
            if not metadata_path.is_file() and baseline_metadata is None:
                raise ValueError("Offline baseline arrays require baseline_metadata.json or explicit baseline_metadata")
            stored = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else baseline_metadata
            if baseline_metadata is not None and stored != baseline_metadata:
                raise ValueError("Supplied baseline_metadata conflicts with baseline_metadata.json")
            self.baseline_metadata = stored
            if not isinstance(self.baseline_metadata, dict):
                raise ValueError("baseline_metadata.json must contain an object")
        if not {"z", "pos"} <= self.arrays.keys():
            raise ValueError("Dataset requires z.npy and pos.npy")
        z, pos = self.arrays["z"], self.arrays["pos"]
        self.ragged = "offsets" in self.arrays
        if self.ragged:
            offsets = self.arrays["offsets"]
            if (pos.ndim != 2 or pos.shape[-1] != 3 or z.shape != pos.shape[:1]
                    or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
                    or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(z)
                    or np.any(offsets[1:] <= offsets[:-1])):
                raise ValueError("Ragged data requires pos[A,3], z[A], increasing offsets[S+1] from 0 to A")
            self.size = len(offsets) - 1
            atomic_shape = (len(z),)
        else:
            if pos.ndim != 3 or pos.shape[-1] != 3 or min(pos.shape[:2]) == 0:
                raise ValueError("Dense pos.npy must have shape [samples, atoms, 3]")
            self.size, n = pos.shape[:2]
            if z.shape not in ((n,), (self.size, n)):
                raise ValueError("Dense z.npy must have shape [atoms] or [samples, atoms]")
            atomic_shape = (self.size, n)
        if z.dtype.kind not in "iu" or np.any((z < 1) | (z > 118)):
            raise ValueError("z.npy must contain integer atomic numbers 1..118; padding is unsupported")
        shapes = {
            "energy": ((self.size,), (self.size, 1)),
            "forces": (pos.shape,), "charges": (atomic_shape, atomic_shape + (1,)),
            "dipole": ((self.size, 3),), "stress": ((self.size, 3, 3),),
            "cell": ((self.size, 3, 3),), "pbc": ((self.size, 3),),
            "charge": ((self.size,), (self.size, 1)), "spin": ((self.size,),),
        }
        for name, allowed in shapes.items():
            if name in self.arrays and self.arrays[name].shape not in allowed:
                raise ValueError(f"{name}.npy shape must be one of {allowed}")
        for name, reference in (("baseline_energy", "energy"), ("baseline_forces", "forces")):
            if name in self.arrays and self.arrays[name].shape not in shapes[reference]:
                raise ValueError(f"{name}.npy has invalid shape")
        self.mm_ragged = "mm_offsets" in self.arrays
        mm_names = set(self.mm_fields) | {"mm_forces", "mm_potential", "mm_potential_gradient", "baseline_mm_forces", "baseline_mm_potential", "baseline_mm_potential_gradient"}
        if mm_names & self.arrays.keys():
            if not {"mm_pos", "mm_charges"} <= self.arrays.keys():
                raise ValueError("MM data requires mm_pos and mm_charges")
            mp = self.arrays["mm_pos"]
            if self.mm_ragged:
                off = self.arrays["mm_offsets"]
                if (mp.ndim != 2 or mp.shape[-1] != 3 or off.shape != (self.size + 1,)
                        or off.dtype.kind not in "iu" or off[0] != 0 or off[-1] != len(mp)
                        or np.any(off[1:] < off[:-1])):
                    raise ValueError("MM ragged data requires mm_pos[M,3] and nondecreasing mm_offsets[S+1]")
            elif mp.ndim != 3 or mp.shape[0] != self.size or mp.shape[-1] != 3:
                raise ValueError("Dense mm_pos must have shape [samples, MM points, 3]")
            for name in ("mm_charges", "mm_potential", "baseline_mm_potential"):
                if name in self.arrays and self.arrays[name].shape != mp.shape[:-1]:
                    raise ValueError(f"{name} must match the MM point layout")
            for name in ("mm_forces", "mm_potential_gradient", "baseline_mm_forces", "baseline_mm_potential_gradient"):
                if name in self.arrays and self.arrays[name].shape != mp.shape:
                    raise ValueError(f"{name} must match mm_pos")
        for name in ("frame_index", "source_index", "short_contact"):
            if name in self.arrays:
                value = self.arrays[name]
                if value.shape != (self.size,) or value.dtype.kind not in "biu":
                    raise ValueError(f"{name} must be a per-frame integer array")
                if name == "short_contact" and not np.isin(value, [0, 1]).all():
                    raise ValueError("short_contact must contain 0/1")
        if "pbc" in self.arrays:
            pbc = self.arrays["pbc"]
            if not np.isin(pbc, [0, 1]).all():
                raise ValueError("pbc.npy must contain booleans or 0/1")
            if np.any(pbc) and "cell" not in self.arrays:
                raise ValueError("Periodic samples require cell.npy")
            if np.any(np.any(pbc, axis=1) != np.all(pbc, axis=1)):
                raise ValueError("Mixed PBC is unsupported")
            periodic = np.all(pbc, axis=1)
            if periodic.any() and np.any(np.abs(np.linalg.det(self.arrays["cell"][periodic])) < 1e-12):
                raise ValueError("Periodic cells must be nonsingular")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        atomic = slice(*self.arrays["offsets"][index:index + 2]) if self.ragged else index
        sample = {}
        for name, values in self.arrays.items():
            if name in ("offsets", "mm_offsets"):
                continue
            key = atomic if name in ("z", "pos", "forces", "charges", "baseline_forces") else index
            if name in ("mm_pos", "mm_charges", "mm_forces", "mm_potential", "mm_potential_gradient", "baseline_mm_forces", "baseline_mm_potential", "baseline_mm_potential_gradient"):
                key = slice(*self.arrays["mm_offsets"][index:index + 2]) if self.mm_ragged else index
            value = values if name == "z" and not self.ragged and values.ndim == 1 else values[key]
            sample[name] = np.array(value, copy=True)
        for name in ("pos", "cell", "mm_pos"):
            if name in sample:
                sample[name] = sample[name] * self.length_scale
        for name, factor in (("energy", self.energy_scale),
                             ("forces", self.energy_scale / self.length_scale * (-1 if self.gradients else 1)),
                             ("stress", self.energy_scale / self.length_scale ** 3)):
            if name in sample:
                sample[name] = sample[name] * factor
        for name, factor in (("mm_forces", self.energy_scale / self.length_scale * (-1 if self.gradients else 1)),
                             ("mm_potential", self.energy_scale),
                             # GESP is already a gradient, not a force label.
                             ("mm_potential_gradient", self.energy_scale / self.length_scale),
                             ("baseline_energy", self.energy_scale),
                             ("baseline_forces", self.energy_scale / self.length_scale),
                             ("baseline_mm_forces", self.energy_scale / self.length_scale),
                             ("baseline_mm_potential", self.energy_scale),
                             ("baseline_mm_potential_gradient", self.energy_scale / self.length_scale)):
            if name in sample:
                sample[name] = sample[name] * factor
        if self.baseline_metadata is not None:
            sample["baseline_metadata"] = dict(self.baseline_metadata)
        return sample

    @staticmethod
    def atoms(sample):
        return Atoms(numbers=sample["z"], positions=sample["pos"],
                     cell=sample.get("cell"), pbc=sample.get("pbc", False))


class NpyShards(Dataset):
    """Join named groups, with an explicit {shard} template for each field.

    Shared fields omit the placeholder. No glob-order matching or silent
    skipping: each requested group must contain all mapped files.
    """

    def __init__(self, directory, shards, *, files, **kwargs):
        shards = list(shards)
        if not shards or len(set(shards)) != len(shards):
            raise ValueError("shards must be a nonempty list of unique group names")
        self.datasets = [NpyDataset(directory, files={k: v.format(shard=s) for k, v in files.items()},
                                    **kwargs) for s in shards]
        self.ends = np.cumsum([len(d) for d in self.datasets])
        self.fields = set(self.datasets[0].arrays)
        fields = self.fields - {"offsets", "mm_offsets"}
        if any(set(d.arrays) - {"offsets", "mm_offsets"} != fields for d in self.datasets):
            raise ValueError("All shards must contain the same sample fields")

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = int(np.searchsorted(self.ends, index, side="right"))
        start = int(self.ends[group - 1]) if group else 0
        return self.datasets[group][index - start]
