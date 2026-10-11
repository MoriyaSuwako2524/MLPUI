"""UMA inference jobs on numeric NPY datasets."""
from pathlib import Path
import copy
import numpy as np

DEFAULT_CONFIG = {"task": "omol", "charge": 0, "spin": 1}


def normalize_prediction(value):
    from mlpui.web.config import dataset
    value = copy.deepcopy(value)
    if value.get("family") != "uma":
        raise ValueError("Prediction currently supports UMA only")
    if set(value) - {"name", "family", "task_type", "model_config", "checkpoint", "prediction", "training"}:
        raise ValueError("Unsupported prediction fields")
    name = value.get("name", "UMA prediction")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise ValueError("Task name must contain 1–100 characters")
    value["name"] = name.strip()
    path = Path(value.get("checkpoint") or "").expanduser().resolve()
    if not path.is_file():
        raise ValueError("UMA prediction requires an existing checkpoint file")
    value["checkpoint"] = str(path)
    config = value.get("model_config", DEFAULT_CONFIG)
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("UMA settings support task, charge and spin only")
    config = {**DEFAULT_CONFIG, **config}
    if config["task"] != "omol":
        raise ValueError("Native UMA prediction currently supports omol only")
    for key, low, high in (("charge", -100, 100), ("spin", 1, 100)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"UMA {key} must be an integer between {low} and {high}")
    value["model_config"] = config
    options = value.get("training", {})
    if not isinstance(options, dict) or set(options) - {"device", "dtype"}:
        raise ValueError("Prediction accepts device and dtype only")
    if options.get("dtype", "float32") != "float32":
        raise ValueError("UMA prediction currently requires float32")
    import re
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", options.get("device", "cpu")):
        raise ValueError("Invalid prediction device")
    value["training"] = options
    spec = value.get("prediction")
    if not isinstance(spec, dict) or not spec.get("directory"):
        raise ValueError("A prediction dataset directory is required")
    if set(spec) - {"directory", "files", "shards", "gradients", "energy_scale", "length_scale", "dataset_id", "dataset_name"}:
        raise ValueError("Unsupported dataset fields")
    spec["directory"] = str(Path(spec["directory"]).expanduser().resolve())
    if "files" in spec and (not isinstance(spec["files"], dict) or
                            not all(isinstance(k, str) and isinstance(v, str) and v for k, v in spec["files"].items())):
        raise ValueError("File mapping must map field names to NPY filenames")
    if "shards" in spec and (not isinstance(spec["shards"], list) or not spec["shards"] or
                             not all(isinstance(s, str) and s for s in spec["shards"])):
        raise ValueError("Groups must be a nonempty list of names")
    # Validate every structure, including per-structure charge and multiplicity.
    for sample in dataset(spec):
        if np.any(sample.get("pbc", False)):
            raise ValueError("Native UMA prediction requires nonperiodic molecules")
        if len(sample["z"]) < 2:
            raise ValueError("UMA WebUI requires at least two atoms per structure; isolated atom references are not configured")
        for key, low, high in (("charge", -100, 100), ("spin", 1, 100)):
            number = np.asarray(sample.get(key, config[key])).item()
            if not np.isfinite(number) or number != int(number) or not low <= number <= high:
                raise ValueError(f"UMA {key} must be an integer between {low} and {high}")
    return value


def build_calculator(checkpoint, task, device):
    """Use the bundled complete predictor, including heads and normalization."""
    import torch
    from mlpui.calculator import CalculatorBuilder
    return CalculatorBuilder.from_checkpoint(checkpoint, family="uma", task=task,
        charge=0, spin=1, device=device, dtype=torch.float32).build()


def run_prediction(settings, folder, device, progress, should_stop):
    from mlpui.web.config import dataset
    from mlpui.training import TrainingStopped
    from mlpui.data import NpyDataset
    data = dataset(settings["prediction"])
    config = settings["model_config"]
    calculator = build_calculator(settings["checkpoint"], config["task"], device)
    energies, forces, offsets = [], [], [0]
    for index, sample in enumerate(data):
        if should_stop():
            raise TrainingStopped()
        atoms = NpyDataset.atoms(sample)
        for key in ("charge", "spin"):
            atoms.info[key] = int(np.asarray(sample.get(key, config[key])).item())
            setattr(calculator, key, atoms.info[key])
        calculator.calculate(atoms, properties=["energy", "forces"])
        energy = float(calculator.results["energy"])
        force = np.asarray(calculator.results["forces"])
        if force.shape != (len(atoms), 3) or not np.isfinite(force).all() or not np.isfinite(energy):
            raise ValueError("UMA produced invalid energy or forces")
        energies.append(energy)
        forces.append(force.copy())
        offsets.append(offsets[-1] + len(atoms))
        progress({"phase": "prediction", "completed": index + 1, "total": len(data)})
    folder = Path(folder)
    np.save(folder / "energy.npy", np.asarray(energies))
    np.save(folder / "forces.npy", np.concatenate(forces))
    np.save(folder / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    return {"samples": len(data), "atoms": offsets[-1], "units": {"energy": "eV", "forces": "eV/angstrom"},
            "files": ["energy.npy", "forces.npy", "offsets.npy"],
            "layout": "forces[offsets[i]:offsets[i+1]] gives forces for structure i"}
