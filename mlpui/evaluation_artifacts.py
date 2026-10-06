"""Validated, atomic NumPy artifacts for standalone evaluations."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from mlpui.data import NpyDataset, NpyShards


ATOMIC_PROPERTIES = {"forces", "charges"}
SPLIT_MASKS = ("is_train", "is_validation", "is_test", "is_guard")


def checkpoint_identity(path):
    """Return a reproducible identity without loading or unpickling a checkpoint."""
    path = Path(path).expanduser().resolve()
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": digest.hexdigest()}


def _parts(data):
    return data.datasets if isinstance(data, NpyShards) else [data]


def _dataset_identity(data):
    result = {"kind": "sharded" if isinstance(data, NpyShards) else "single",
              "samples": len(data), "parts": []}
    for index, part in enumerate(_parts(data)):
        files = {}
        for field, array in part.arrays.items():
            path = Path(array.filename).resolve()
            files[field] = {"path": str(path), "bytes": path.stat().st_size,
                            "shape": list(array.shape), "dtype": str(array.dtype)}
        result["parts"].append({"index": index, "directory": str(part.directory.resolve()),
                                "samples": len(part), "files": files})
    return result


def _source_rows(data):
    if isinstance(data, NpyShards):
        groups = np.concatenate([np.full(len(part), group, dtype=np.int32)
                                 for group, part in enumerate(data.datasets)])
        rows = np.concatenate([np.arange(len(part), dtype=np.int64) for part in data.datasets])
        return groups, rows
    return np.zeros(len(data), dtype=np.int32), np.arange(len(data), dtype=np.int64)


def _load_sidecar(data, name):
    """Load a global sidecar whose rows match the exact evaluation order."""
    parts = _parts(data)
    directories = {part.directory.resolve() for part in parts}
    if len(directories) != 1:
        return None
    path = next(iter(directories)) / f"{name}.npy"
    if not path.is_file():
        return None
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if (not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != len(data)
            or value.dtype.kind not in "biuf" or not np.isfinite(value).all()):
        raise ValueError(f"{name}.npy must be a finite numeric one-dimensional array with {len(data)} rows")
    return np.asarray(value)


def _validated_sidecars(data):
    frame_index = _load_sidecar(data, "frame_index")
    if frame_index is None:
        frame_index = np.arange(len(data), dtype=np.int64)
        frame_origin = "generated_evaluation_order"
    else:
        if frame_index.dtype.kind not in "iu":
            raise ValueError("frame_index.npy must contain integers")
        if len(np.unique(frame_index)) != len(frame_index):
            raise ValueError("frame_index.npy must contain unique frame identifiers")
        frame_index = frame_index.astype(np.int64, copy=False)
        frame_origin = "dataset_sidecar"

    raw_masks = {name: _load_sidecar(data, name) for name in SPLIT_MASKS}
    present = [name for name, value in raw_masks.items() if value is not None]
    if present and len(present) != len(SPLIT_MASKS):
        missing = sorted(set(SPLIT_MASKS) - set(present))
        raise ValueError(f"Split sidecars are incomplete; missing {missing}")
    masks = {}
    if present:
        for name, value in raw_masks.items():
            if value.dtype.kind not in "biu" or not np.isin(value, [0, 1]).all():
                raise ValueError(f"{name}.npy must contain booleans or 0/1 integers")
            masks[name] = value.astype(bool, copy=False)
        if not np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1):
            raise ValueError("Split sidecars must be mutually exclusive and cover every evaluated sample")
    return frame_index, frame_origin, masks


def _canonical_sample(value, prop):
    value = np.asarray(value, dtype=np.float64)
    if prop == "energy" and value.size == 1:
        value = value.reshape(())
    elif prop == "charges" and value.ndim > 1 and value.shape[-1] == 1:
        value = value.reshape(value.shape[:-1])
    if value.dtype.kind not in "f" or not np.isfinite(value).all():
        raise ValueError(f"Non-finite or non-numeric {prop} evaluation artifact")
    return value


def _combine(values, prop, atom_counts, variable_atoms):
    values = [_canonical_sample(value, prop) for value in values]
    if prop in ATOMIC_PROPERTIES:
        for index, (value, atoms) in enumerate(zip(values, atom_counts)):
            if value.ndim == 0 or value.shape[0] != atoms:
                raise ValueError(f"{prop} artifact row {index} does not match its atom count")
        if variable_atoms:
            return np.concatenate(values, axis=0), "flat_atoms"
    try:
        return np.stack(values), "dense"
    except ValueError as exc:
        raise ValueError(f"Inconsistent per-sample shapes for {prop}") from exc


def _write_npy(path, value):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"Refusing to save non-numeric or non-finite artifact {path.name}")
    np.save(path, value, allow_pickle=False)


def save_evaluation_artifacts(observations, data, directory, *, metadata=None):
    """Publish prediction/reference arrays only after the complete set validates.

    ``observations`` maps each property to ordered ``(reference, prediction)``
    pairs, one pair per evaluated structure.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Evaluation artifact directory already exists: {directory}")
    staging = directory.with_name(f".{directory.name}.{uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    try:
        atom_counts = np.asarray([len(data[index]["z"]) for index in range(len(data))], dtype=np.int64)
        variable_atoms = bool(len(np.unique(atom_counts)) > 1 or any(part.ragged for part in _parts(data)))
        files, properties = {}, {}
        for prop, pairs in observations.items():
            if len(pairs) != len(data):
                raise ValueError(f"{prop} observations do not cover every evaluated sample")
            references, predictions = zip(*pairs)
            reference, ref_layout = _combine(references, prop, atom_counts, variable_atoms)
            prediction, pred_layout = _combine(predictions, prop, atom_counts, variable_atoms)
            if reference.shape != prediction.shape or ref_layout != pred_layout:
                raise ValueError(f"Reference and prediction artifact shapes differ for {prop}")
            ref_name, pred_name = f"{prop}_ref.npy", f"{prop}_pred.npy"
            _write_npy(staging / ref_name, reference)
            _write_npy(staging / pred_name, prediction)
            files[ref_name] = {"role": "dataset_reference_label", "shape": list(reference.shape),
                               "dtype": str(reference.dtype)}
            files[pred_name] = {"role": "model_prediction", "shape": list(prediction.shape),
                                "dtype": str(prediction.dtype)}
            properties[prop] = {"layout": ref_layout, "reference": ref_name, "prediction": pred_name,
                                "unit": "e" if prop == "charges" else "converted_dataset_units"}

        source_group, source_row = _source_rows(data)
        frame_index, frame_origin, masks = _validated_sidecars(data)
        ordered = {"frame_index.npy": frame_index, "source_row_index.npy": source_row}
        if isinstance(data, NpyShards):
            ordered["source_group.npy"] = source_group
        ordered.update({f"{name}.npy": value for name, value in masks.items()})
        if variable_atoms:
            ordered["structure_offsets.npy"] = np.concatenate(([0], np.cumsum(atom_counts)))
        for name, value in ordered.items():
            _write_npy(staging / name, value)
            files[name] = {"role": name.removesuffix(".npy"), "shape": list(value.shape),
                           "dtype": str(value.dtype)}

        manifest = {
            "schema_version": 1,
            "samples": len(data),
            "ordering": "sequential NpyDataset/NpyShards iteration; no shuffle",
            "frame_index_origin": frame_origin,
            "atom_layout": "flat_with_structure_offsets" if variable_atoms else "dense",
            "properties": properties,
            "files": files,
            "reference_semantics": "Reference arrays are labels supplied by the evaluation dataset; prediction arrays are model outputs.",
        }
        manifest["provenance"] = {"dataset": _dataset_identity(data), **(metadata or {})}
        manifest_path = staging / "metadata.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2),
                                 encoding="utf-8")
        with manifest_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        staging.replace(directory)
        return {"directory": directory.name, "manifest": "metadata.json", "files": files,
                "properties": properties, "frame_index_origin": frame_origin}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
