"""Numeric .npy datasets shared by all training backends (no pickle)."""
from pathlib import Path

import numpy as np
from ase import Atoms
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    """A directory of z/pos/target arrays, optionally cell/pbc/charge/spin.

    Dense: pos[S,N,3], z[N] or z[S,N]. Ragged: pos[A,3], z[A],
    offsets[S+1]. Atomic targets follow the positions layout. All samples
    use the same target fields; units must already match the model.
    """

    targets = ("energy", "forces", "charges", "dipole", "stress")

    def __init__(self, directory, *, files=None, gradients=False, energy_scale=1.0,
                 length_scale=1.0):
        self.directory = Path(directory)
        self.gradients = gradients
        self.energy_scale, self.length_scale = energy_scale, length_scale
        if not all(np.isfinite(v) and v > 0 for v in (energy_scale, length_scale)):
            raise ValueError("Unit scales must be finite and positive")
        names = ("z", "pos", "offsets", "cell", "pbc", "charge", "spin") + self.targets
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
            if name == "offsets":
                continue
            key = atomic if name in ("z", "pos", "forces", "charges") else index
            value = values if name == "z" and not self.ragged and values.ndim == 1 else values[key]
            sample[name] = np.array(value, copy=True)
        for name in ("pos", "cell"):
            if name in sample:
                sample[name] = sample[name] * self.length_scale
        for name, factor in (("energy", self.energy_scale),
                             ("forces", self.energy_scale / self.length_scale * (-1 if self.gradients else 1)),
                             ("stress", self.energy_scale / self.length_scale ** 3)):
            if name in sample:
                sample[name] = sample[name] * factor
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
