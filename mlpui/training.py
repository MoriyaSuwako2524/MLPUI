"""Unified .npy training interface for the optional TorchMD-Net/NewtonNet forks."""
from dataclasses import dataclass, field
from pathlib import Path
import copy
import math
import os
import re
import time
from uuid import uuid4

import numpy as np
import torch

from mlpui.data import NpyDataset, NpyShards
from mlpui.external_calculator import AtomicInputAdapter, AtomicOutputAdapter
from mlpui.external_models import _config_dict, load_external_checkpoint


class TrainingStopped(Exception):
    """Cooperative cancellation at a structure boundary."""


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-3
    loss_weights: dict = field(default_factory=lambda: {"energy": 1.0, "forces": 1.0})
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    save_interval: int = 0
    max_checkpoints: int = 3
    test_interval: int = 1

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (self.epochs, self.batch_size)):
            raise ValueError("epochs and batch_size must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (not self.loss_weights or set(self.loss_weights) - set(NpyDataset.targets)
                or any(not math.isfinite(w) or w <= 0 for w in self.loss_weights.values())):
            raise ValueError("loss_weights requires supported targets with finite positive weights")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Training supports float32 and float64")
        for name, minimum in (("save_interval", 0), ("max_checkpoints", 1), ("test_interval", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class Trainer:
    """Create a fresh model or fine-tune a checkpoint, using only .npy datasets.

    Batches accumulate per-structure gradients to support differing atom counts
    and cells on both backends. Loss is component-mean MSE, averaged over
    structures, then summed with the configured property weights.
    """

    def __init__(self, family, model_config, config=None, *, checkpoint=None,
                 trusted_checkpoint=False):
        self.family = {"torchmd-net": "torchmdnet", "tensornet": "torchmdnet"}.get(family, family)
        if self.family not in ("torchmdnet", "newtonnet"):
            raise ValueError("Training currently supports torchmdnet and newtonnet")
        self.config = config or TrainingConfig()
        self.model_config = _config_dict(model_config)
        if self.family == "newtonnet" and isinstance(self.model_config.get("model"), dict):
            self.model_config = self.model_config["model"]
        torch.manual_seed(self.config.seed)
        if self.family == "torchmdnet":
            self.model_config["precision"] = 64 if self.config.dtype == torch.float64 else 32
            if "forces" in self.config.loss_weights:
                self.model_config["derivative"] = True
        if checkpoint is not None:
            self.model = load_external_checkpoint(
                checkpoint, family=self.family, model_config=self.model_config,
                device=self.config.device, dtype=self.config.dtype,
                trusted_checkpoint=trusted_checkpoint).model
        elif self.family == "torchmdnet":
            from torchmdnet.models.model import create_model
            self.model = create_model(copy.deepcopy(self.model_config))
        else:
            from newtonnet.models.newtonnet import NewtonNet
            self.model = NewtonNet(**self.model_config)
        self.model.to(device=self.config.device, dtype=self.config.dtype)
        if self.family == "torchmdnet":
            if self.model_config.get("output_model", "Scalar") != "Scalar":
                raise ValueError("Training requires a Scalar primary energy head")
            if "forces" in self.config.loss_weights:
                self.model.derivative = True
        elif set(self.model.output_properties).intersection({"hessian", "bec"}):
            raise ValueError("Hessian/BEC heads require specialized training")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.history = []

    def _loss(self, sample):
        atoms = NpyDataset.atoms(sample)
        if "stress" in self.config.loss_weights and not np.all(atoms.pbc):
            raise ValueError("Stress training requires periodic structures")
        inputs = AtomicInputAdapter(self.family, charge=float(sample.get("charge", 0)),
                                    spin=float(sample.get("spin", 0))).convert(
                                        atoms, self.config.device, self.config.dtype)
        if self.family == "torchmdnet":
            for module in self.model.modules():
                if type(module).__name__ == "OptimizedDistance":
                    module.use_periodic = inputs["box"] is not None
        output = self.model(**inputs)
        if isinstance(output, tuple):
            output = dict(zip(("energy", "forces"), output))
        elif not isinstance(output, dict):
            output = vars(output)
        losses = {}
        for prop in self.config.loss_weights:
            prediction = next((output[k] for k in AtomicOutputAdapter.ALIASES[prop]
                               if output.get(k) is not None), None)
            if prediction is None:
                raise ValueError(f"Model does not produce training target {prop}")
            target = torch.as_tensor(sample[prop], device=self.config.device, dtype=self.config.dtype)
            if prediction.numel() != target.numel():
                raise ValueError(f"Model output shape {tuple(prediction.shape)} does not match {prop} target {tuple(target.shape)}")
            losses[prop] = (prediction.reshape(target.shape) - target).square().mean()
        total = sum(self.config.loss_weights[k] * v for k, v in losses.items())
        if not torch.isfinite(total):
            raise ValueError("Non-finite training loss")
        return total, losses

    def fit(self, train_data, validation_data=None, *, test_data=None, output_dir=None,
            on_progress=None, should_stop=None):
        def check_stop():
            if should_stop is not None and should_stop():
                raise TrainingStopped("Training stopped by user")

        check_stop()
        if self.config.save_interval and output_dir is None:
            raise ValueError("Periodic saving requires output_dir")
        train = train_data if isinstance(train_data, (NpyDataset, NpyShards)) else NpyDataset(train_data)
        validation = (validation_data if isinstance(validation_data, (NpyDataset, NpyShards)) else
                      NpyDataset(validation_data)) if validation_data is not None else None
        test = (test_data if isinstance(test_data, (NpyDataset, NpyShards)) else
                NpyDataset(test_data)) if test_data is not None else None
        for dataset in (train, validation, test):
            if dataset is not None:
                fields = dataset.fields if isinstance(dataset, NpyShards) else dataset.arrays.keys()
                missing = set(self.config.loss_weights) - fields
                if missing:
                    raise ValueError(f"Dataset missing labels: {sorted(missing)}")
        rng = np.random.default_rng(self.config.seed)
        for _ in range(self.config.epochs):
            check_stop()
            self.model.train()
            sums = {key: 0.0 for key in self.config.loss_weights}
            order = rng.permutation(len(train))
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start:start + self.config.batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                for index in indices:
                    check_stop()
                    with torch.enable_grad():
                        loss, parts = self._loss(train[int(index)])
                        (loss / len(indices)).backward()
                    for key, value in parts.items():
                        sums[key] += value.detach().item()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError("Non-finite parameter gradient")
                self.optimizer.step()
                if on_progress is not None:
                    on_progress({"phase": "training", "epoch": len(self.history) + 1,
                                 "completed": min(start + len(indices), len(train)),
                                 "total": len(train), "history": self.history})
            row = {"epoch": len(self.history) + 1, "train": {k: v / len(train) for k, v in sums.items()}}
            evaluations = [("validation", validation)]
            if row["epoch"] % self.config.test_interval == 0:
                evaluations.append(("test", test))
            for split, evaluation in evaluations:
                if evaluation is None:
                    continue
                if on_progress is not None:
                    on_progress({"phase": split, "epoch": len(self.history) + 1,
                                 "completed": len(train), "total": len(train), "history": self.history})
                self.model.eval()
                sums = {key: 0.0 for key in self.config.loss_weights}
                # Derivative heads still need autograd during validation.
                with torch.enable_grad():
                    for sample in evaluation:
                        check_stop()
                        _, parts = self._loss(sample)
                        for key, value in parts.items():
                            sums[key] += value.detach().item()
                row[split] = {k: v / len(evaluation) for k, v in sums.items()}
            self.history.append(row)
            if self.config.save_interval and row["epoch"] % self.config.save_interval == 0:
                self._save_periodic(output_dir, row["epoch"])
            if on_progress is not None:
                on_progress({"phase": "epoch_end", "epoch": len(self.history),
                             "completed": len(train), "total": len(train),
                             "history": self.history})
        self.model.eval()
        if output_dir is not None:
            self.save(Path(output_dir) / "model.pt")
        return self.history

    def _save_periodic(self, output_dir, epoch):
        directory = Path(output_dir) / "checkpoints"
        self.save(directory / f"epoch_{epoch:06d}.pt")
        # Only prune this interface's exact checkpoint names, after saving succeeds.
        saved = sorted((p for p in directory.iterdir()
                        if p.is_file() and re.fullmatch(r"epoch_[0-9]{6,}\.pt", p.name)),
                       key=lambda p: int(p.stem.split("_")[1]))
        for path in saved[:-self.config.max_checkpoints]:
            path.unlink()

    def save(self, path):
        """Export portable weights/config for CalculatorBuilder (not optimizer resume)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            torch.save({"model_config": self.model_config,
                    "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "history": self.history, "epoch": len(self.history)}, temporary)
            for attempt in range(50):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 49:
                        raise
                    time.sleep(.02)
        finally:
            temporary.unlink(missing_ok=True)
        return path
