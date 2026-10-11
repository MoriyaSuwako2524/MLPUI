"""Persistent dataset catalog, streamed NPY uploads and reproducible splits."""
import copy
from pathlib import Path
import re
import shutil
import threading
import time
from uuid import uuid4

import numpy as np

from mlpui.data import NpyShards
from mlpui.web.config import dataset
from mlpui.web.worker import read_json, write_json


def checked_spec(spec):
    spec = copy.deepcopy(spec)
    if not isinstance(spec, dict) or not spec.get("directory"):
        raise ValueError("Enter a data directory")
    if spec.keys() - {"directory", "files", "shards", "gradients", "energy_scale", "length_scale"}:
        raise ValueError("Unsupported dataset fields")
    spec["directory"] = str(Path(spec["directory"]).expanduser().resolve())
    if "files" in spec and (not isinstance(spec["files"], dict) or
            not all(isinstance(k, str) and isinstance(v, str) and v for k, v in spec["files"].items())):
        raise ValueError("File mapping must map field names to filenames")
    if "shards" in spec and (not isinstance(spec["shards"], list) or
            not all(isinstance(v, str) and v for v in spec["shards"])):
        raise ValueError("Groups must be a list of nonempty names")
    if "gradients" in spec and not isinstance(spec["gradients"], bool):
        raise ValueError("gradients must be boolean")
    return spec


def inspect_spec(spec):
    data = dataset(spec)
    parts = data.datasets if isinstance(data, NpyShards) else [data]
    fields = set(parts[0].arrays) - {"offsets"}
    if any(set(part.arrays) - {"offsets"} != fields for part in parts):
        raise ValueError("All groups must contain the same fields")
    fields = set().union(*(part.arrays.keys() for part in parts))
    files = []
    for i, part in enumerate(parts):
        for key, array in part.arrays.items():
            path = Path(array.filename)
            files.append(dict(field=key, group=spec.get("shards", [""])[i],
                              filename=path.name, shape=list(array.shape), dtype=str(array.dtype),
                              bytes=path.stat().st_size))
    return data, dict(samples=len(data), groups=len(parts), fields=sorted(fields), files=files)


def export_subset(data, indices, directory):
    """Write converted samples as numeric ragged NPY arrays with bounded RAM."""
    directory = Path(directory)
    directory.mkdir()
    parts = data.datasets if isinstance(data, NpyShards) else [data]
    all_sizes = np.concatenate([np.diff(part.arrays["offsets"]) if part.ragged
                                else np.full(len(part), part.arrays["pos"].shape[1], dtype=np.int64)
                                for part in parts])
    sizes = all_sizes[indices]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    np.save(directory / "offsets.npy", offsets)
    first = data[int(indices[0])]
    atomic = {"z", "pos", "forces", "charges"}
    # Memory-map each output and copy a single structure at a time.
    outputs = {}
    try:
        for key, initial in first.items():
            initial = np.asarray(initial)
            if key == "charges":
                initial = initial.reshape(-1)
            if key in ("energy", "charge", "spin"):
                initial = initial.reshape(())
            shape = ((int(offsets[-1]),) + initial.shape[1:] if key in atomic
                     else (len(indices),) + initial.shape)
            dtype = np.result_type(initial.dtype, *(part.arrays[key].dtype for part in parts))
            outputs[key] = np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
        for row, index in enumerate(indices):
            for key, value in data[int(index)].items():
                if key == "charges":
                    value = value.reshape(-1)
                if key in ("energy", "charge", "spin"):
                    value = value.reshape(())
                outputs[key][slice(offsets[row], offsets[row + 1]) if key in atomic else row] = value
        for output in outputs.values():
            output.flush()
    finally:
        outputs.clear()
    return {"directory": str(directory), "files": {k: f"{k}.npy" for k in (*first, "offsets")},
            "gradients": False, "energy_scale": 1., "length_scale": 1.}


class DatasetManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def folder(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid dataset ID")
        return self.root / identifier

    def get(self, identifier):
        return read_json(self.folder(identifier) / "dataset.json")

    def list(self):
        with self.lock:
            return sorted([read_json(p) for p in self.root.glob("*/dataset.json")],
                          key=lambda item: item["created"], reverse=True)

    @staticmethod
    def tags(values):
        if not isinstance(values, list) or len(values) > 20:
            raise ValueError("At most 20 tags are allowed")
        result = []
        for value in values:
            if not isinstance(value, str) or not 1 <= len(value.strip()) <= 40:
                raise ValueError("Each tag must contain 1–40 characters")
            value = value.strip()
            if value.casefold() not in {v.casefold() for v in result}:
                result.append(value)
        return result

    @staticmethod
    def name(value):
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 100:
            raise ValueError("Dataset name must contain 1–100 characters")
        return value.strip()

    def create(self, name, spec=None, tags=None):
        with self.lock:
            return self._create(name, spec, tags)

    def _create(self, name, spec, tags):
        name = self.name(name)
        tags = self.tags(tags if tags is not None else [])
        if spec is not None:
            spec = checked_spec(spec)
            _, summary = inspect_spec(spec)
        with self.lock:
            identifier = uuid4().hex
            folder = self.folder(identifier)
            folder.mkdir()
            record = dict(id=identifier, name=name, tags=tags, created=time.time(), archived=False,
                          status="ready" if spec is not None else "uploading", source="existing" if spec else "upload")
            if spec is not None:
                record.update(spec=spec, summary=summary)
            else:
                (folder / "data").mkdir()
            write_json(folder / "dataset.json", record)
            return record

    def upload(self, identifier, filename, stream, size):
        # Never interpret client filenames as paths, including Windows path syntax.
        if (not isinstance(filename, str) or len(filename) > 200 or
                not re.fullmatch(r"[\w .-]+\.npy", filename) or filename.startswith(".")):
            raise ValueError("Use a plain .npy filename without directory paths")
        if not 0 < size <= 20 * 1024**3:
            raise ValueError("Each upload must be between 1 byte and 20 GiB")
        with self.lock:
            record = self.get(identifier)
            if record["status"] != "uploading" or record["archived"]:
                raise ValueError("This dataset does not accept further uploads")
            directory = self.folder(identifier) / "data"
            destination = directory / filename
            if destination.exists():
                raise ValueError("Filename already exists; create a new upload dataset")
            temporary = directory / (uuid4().hex + ".part")
            try:
                with temporary.open("xb") as output:
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Upload incomplete; retry")
                        output.write(chunk)
                        remaining -= len(chunk)
                try:
                    value = np.load(temporary, allow_pickle=False, mmap_mode="r")
                    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                        if hasattr(value, "close"):
                            value.close()
                        raise ValueError("Only real numeric NPY arrays without pickle are accepted")
                    del value
                except Exception as exc:
                    raise ValueError("Invalid numeric NPY file") from exc
                temporary.rename(destination)
            finally:
                temporary.unlink(missing_ok=True)
            return {"filename": filename, "bytes": size}

    def finalize(self, identifier, spec):
        with self.lock:
            record = self.get(identifier)
            if record["status"] != "uploading" or record["archived"]:
                raise ValueError("Dataset is not an upload draft")
            directory = self.folder(identifier) / "data"
            spec = checked_spec({**spec, "directory": str(directory)})
            for template in spec.get("files", {}).values():
                for shard in spec.get("shards", [""]):
                    if not (directory / template.format(shard=shard)).resolve().is_relative_to(directory):
                        raise ValueError("Uploads may only reference files in this dataset")
            _, summary = inspect_spec(spec)
            record.update(status="ready", spec=spec, summary=summary)
            write_json(self.folder(identifier) / "dataset.json", record)
            return record

    def update(self, identifier, changes):
        if not isinstance(changes, dict) or changes.keys() - {"name", "archived", "tags"}:
            raise ValueError("Only name, tags and archive status can be changed")
        with self.lock:
            record = self.get(identifier)
            if "name" in changes:
                record["name"] = self.name(changes["name"])
            if "tags" in changes:
                record["tags"] = self.tags(changes["tags"])
            if "archived" in changes:
                if not isinstance(changes["archived"], bool):
                    raise ValueError("archived must be boolean")
                record["archived"] = changes["archived"]
            write_json(self.folder(identifier) / "dataset.json", record)
            return record

    def inspect(self, identifier):
        with self.lock:
            return self._inspect(identifier)

    def _inspect(self, identifier):
        record = self.get(identifier)
        if record["status"] != "ready":
            raise ValueError("Complete upload and validation first")
        _, summary = inspect_spec(record["spec"])
        with self.lock:
            record = self.get(identifier)
            record["summary"] = summary
            write_json(self.folder(identifier) / "dataset.json", record)
        return summary

    def delete(self, identifier, options, *, job_specs=()):
        if not isinstance(options, dict) or set(options) - {"delete_files"} or not isinstance(options.get("delete_files", False), bool):
            raise ValueError("delete_files must be boolean")
        with self.lock:
            record = self.get(identifier)
            if options.get("delete_files", False):
                if record["source"] == "upload":
                    target = self.folder(identifier) / "data"
                elif record["source"] == "split":
                    if not re.fullmatch(r"[a-f0-9]{32}", record["split_id"]) or record["split_role"] not in {"train", "validation", "test"}:
                        raise ValueError("Invalid managed dataset path")
                    target = self.root / ("split-" + record["split_id"]) / record["split_role"]
                else:
                    raise ValueError("Existing directories allow catalog deletion only, not source file deletion")
                resolved = target.resolve()
                if resolved != target or not resolved.is_relative_to(self.root) or resolved == self.root:
                    raise ValueError("Refusing to delete files or links outside managed directories")
                if record.get("spec") and Path(record["spec"]["directory"]).resolve() != resolved:
                    raise ValueError("Dataset path does not match owned directory")

                def references(spec):
                    directory = Path(spec["directory"]).resolve()
                    paths = [directory]
                    for filename in spec.get("files", {}).values():
                        paths.extend((directory / filename.format(shard=s)).resolve() for s in spec.get("shards", [""]))
                    return any(p.is_relative_to(resolved) or resolved.is_relative_to(p) for p in paths)

                if any(references(spec) for spec in job_specs):
                    raise ValueError("Files are referenced by jobs; only the catalog entry can be removed")
                if any(other["id"] != identifier and other.get("spec") and references(other["spec"]) for other in self.list()):
                    raise ValueError("Files are referenced by other datasets; only the catalog entry can be removed")
                if target.exists():
                    # Check every resolved descendant before recursive deletion on Windows.
                    if any(not p.resolve().is_relative_to(resolved) or p.is_symlink() for p in target.rglob("*")):
                        raise ValueError("Data directory contains links; refusing recursive deletion")
                    shutil.rmtree(target)
            # Keep a provenance tombstone while removing the entry from the catalog.
            record.update(deleted=time.time(), files_deleted=options.get("delete_files", False))
            write_json(self.folder(identifier) / "deleted.json", record)
            (self.folder(identifier) / "dataset.json").unlink()
            return {"id": identifier, "deleted": True, "files_deleted": record["files_deleted"]}

    def split(self, identifier, options):
        with self.lock:
            return self._split(identifier, options)

    def _split(self, identifier, options):
        source = self.get(identifier)
        if source["status"] != "ready" or source["archived"]:
            raise ValueError("Select a ready dataset")
        name = self.name(options.get("name"))
        seed = options.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer from 0 to 2^32-1")
        method = options.get("method", "random")
        if method not in ("random", "ordered"):
            raise ValueError("Split method must be random or ordered")
        ratios = np.asarray(options.get("ratios", [80, 10, 10]), dtype=float)
        if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios < 0) or ratios[0] <= 0 or not 0 < ratios.sum() < np.inf:
            raise ValueError("Supply train, validation and test ratios; training must be positive")
        data, summary = inspect_spec(source["spec"])
        exact = ratios / ratios.sum() * len(data)
        counts = np.floor(exact).astype(int)
        for index in np.argsort(-(exact - counts), kind="stable")[:len(data) - counts.sum()]:
            counts[index] += 1
        if np.any((ratios > 0) & (counts == 0)):
            raise ValueError("Too few samples for nonempty subsets; adjust ratios")
        indices = np.random.default_rng(seed).permutation(len(data)) if method == "random" else np.arange(len(data))
        split_id = uuid4().hex
        directory = self.root / ("split-" + split_id)
        directory.mkdir()
        records, start = [], 0
        for label, count in zip(("train", "validation", "test"), counts):
            if not count:
                continue
            selected = indices[start:start + count]
            start += count
            spec = export_subset(data, selected, directory / label)
            np.save(directory / f"{label}_indices.npy", selected)
            _, child_summary = inspect_spec(spec)
            records.append(dict(id=uuid4().hex, name=f"{name} · {label}", created=time.time(),
                                tags=list(source.get("tags", [])),
                                archived=False, status="ready", source="split", spec=spec, summary=child_summary,
                                parent_id=identifier, split_id=split_id, split_role=label))
        manifest = dict(id=split_id, parent_id=identifier, source_spec=source["spec"], samples=len(data),
                        seed=seed, method=method, ratios=ratios.tolist(), counts=counts.tolist(),
                        datasets=[record["id"] for record in records])
        write_json(directory / "split.json", manifest)
        with self.lock:
            for record in records:
                folder = self.folder(record["id"])
                folder.mkdir()
                write_json(folder / "dataset.json", record)
        return {"datasets": records, "manifest": manifest}
