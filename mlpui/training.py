"""Unified .npy training interface for registered model backends."""
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
from mlpui.external_models import _config_dict, load_external_checkpoint
from mlpui.backends import get_backend


class TrainingStopped(Exception):
    """Cooperative cancellation at a structure boundary."""


def configure_charge_head(family, model_config, targets):
    """Enable an independent atomic-charge output when charge labels are selected.

    Existing heads and their order are preserved. A checkpoint must already
    contain these learned parameters; loading remains strict.
    """
    return get_backend(family).configure_targets(model_config, targets)


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
    early_stopping: bool = False
    early_stopping_monitor: str = "loss"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    # Validate every N epochs (and always after the final configured epoch).
    # Early-stopping patience counts validation checks, not epochs.
    validation_interval: int = 1
    # Structures per forward pass during validation/evaluation; None reuses batch_size.
    eval_batch_size: int | None = None
    # Concatenate a batch into one graph for backends that support it. False keeps
    # the per-structure gradient accumulation path (identical loss definition).
    batched: bool = True

    def __post_init__(self):
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be boolean")
        if not isinstance(self.batched, bool):
            raise ValueError("batched must be boolean")
        if self.eval_batch_size is not None and (
                isinstance(self.eval_batch_size, bool) or not isinstance(self.eval_batch_size, int)
                or self.eval_batch_size < 1):
            raise ValueError("eval_batch_size must be a positive integer or null")
        if (isinstance(self.early_stopping_patience, bool) or not isinstance(self.early_stopping_patience, int)
                or self.early_stopping_patience < 1):
            raise ValueError("early_stopping_patience must be a positive integer")
        if (isinstance(self.early_stopping_min_delta, bool) or not isinstance(self.early_stopping_min_delta, (float, int))
                or not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0):
            raise ValueError("early_stopping_min_delta must be finite and nonnegative")
        if self.early_stopping_monitor != "loss" and self.early_stopping_monitor not in self.loss_weights:
            raise ValueError("Early stopping monitor must be loss or an enabled target")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (self.epochs, self.batch_size)):
            raise ValueError("epochs and batch_size must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (not self.loss_weights or set(self.loss_weights) - set(NpyDataset.targets)
                or any(not math.isfinite(w) or w <= 0 for w in self.loss_weights.values())):
            raise ValueError("loss_weights requires supported targets with finite positive weights")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Training supports float32 and float64")
        for name, minimum in (("save_interval", 0), ("max_checkpoints", 1), ("test_interval", 1),
                              ("validation_interval", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class Trainer:
    """Create a fresh model or fine-tune a checkpoint, using only .npy datasets.

    Loss is component-mean MSE within each structure, averaged over the
    structures of a batch, then summed with the configured property weights.
    Backends with ``supports_batching`` evaluate a batch as one concatenated
    graph (``TrainingConfig.batched``); others, or batches a backend declines
    (for example mixed periodic cells), accumulate per-structure gradients.
    Both paths define the same loss and gradient.
    """

    PER_STRUCTURE_TARGETS = ("energy", "dipole", "stress")
    ATOMIC_TARGETS = ("forces", "charges")
    MM_TARGETS = ("mm_forces", "mm_potential", "mm_potential_gradient", "gas_mm_potential", "gas_mm_potential_gradient")

    def __init__(self, family, model_config, config=None, *, checkpoint=None,
                 trusted_checkpoint=False):
        self.backend = get_backend(family)
        self.family = self.backend.name
        self.config = config or TrainingConfig()
        self.model_config = self.backend.training_config(
            _config_dict(model_config), self.config.loss_weights, self.config.dtype)
        torch.manual_seed(self.config.seed)
        if checkpoint is not None:
            self.model = load_external_checkpoint(
                checkpoint, family=self.family, model_config=self.model_config,
                device=self.config.device, dtype=self.config.dtype,
                trusted_checkpoint=trusted_checkpoint).model
        else:
            self.model = self.backend.build_model(self.model_config)
        self.model.to(device=self.config.device, dtype=self.config.dtype)
        if getattr(self.model, "charge_constraint", False):
            self.model_config["charge_constraint"] = True
        self.backend.validate_training(self.model, self.model_config, self.config.loss_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.history = []

    def _check_sample(self, sample):
        missing = set(self.config.loss_weights) - sample.keys()
        if missing:
            raise ValueError(f"Sample missing enabled labels: {sorted(missing)}")
        if "stress" in self.config.loss_weights and not np.all(NpyDataset.atoms(sample).pbc):
            raise ValueError("Stress training requires periodic structures")

    def _errors(self, sample, on_prediction=None):
        self._check_sample(sample)
        inputs = self.backend.prepare_sample(
            sample, self.config.device, self.config.dtype, model=self.model)
        output = self.backend.forward(self.model, inputs, self.config.loss_weights)
        errors = {}
        for prop in self.config.loss_weights:
            prediction = output.get(prop)
            if prediction is None:
                raise ValueError(f"Model does not produce training target {prop}")
            target = torch.as_tensor(sample[prop], device=self.config.device, dtype=self.config.dtype)
            if prediction.numel() != target.numel():
                raise ValueError(f"Model output shape {tuple(prediction.shape)} does not match {prop} target {tuple(target.shape)}")
            errors[prop] = prediction.reshape(target.shape) - target
            if not torch.isfinite(errors[prop]).all():
                raise ValueError(f"Non-finite prediction or target for {prop}")
            if on_prediction is not None:
                on_prediction(prop, target.detach().cpu().double().numpy(),
                              prediction.reshape(target.shape).detach().cpu().double().numpy())
        return errors

    def _batched_errors(self, samples, on_prediction=None):
        """Per-structure error dicts for several samples from ONE forward pass.

        Returns None when batching is disabled/unsupported for these samples;
        callers then use the per-structure path. Error tensors are views of the
        batched prediction, so per-structure losses keep their autograd graph.
        """
        if (not self.config.batched or len(samples) < 2
                or not getattr(self.backend, "supports_batching", False)):
            return None
        for sample in samples:
            self._check_sample(sample)
        prepared = self.backend.prepare_batch(
            samples, self.config.device, self.config.dtype, model=self.model)
        if prepared is None:
            return None
        inputs, layout = prepared
        output = self.backend.forward(self.model, inputs, self.config.loss_weights)
        sizes = {"structure": [1] * len(samples), "atoms": list(layout["atoms"]),
                 "mm": None if layout.get("mm") is None else list(layout["mm"])}
        errors = [{} for _ in samples]
        for prop in self.config.loss_weights:
            prediction = output.get(prop)
            if prediction is None:
                raise ValueError(f"Model does not produce training target {prop}")
            kind = ("structure" if prop in self.PER_STRUCTURE_TARGETS else
                    "atoms" if prop in self.ATOMIC_TARGETS else
                    "mm" if prop in self.MM_TARGETS else None)
            if kind is None or sizes[kind] is None:
                raise ValueError(f"Batched layout does not describe target {prop}")
            counts = sizes[kind]
            total = sum(counts)
            if total == 0:
                if prediction.numel():
                    raise ValueError(f"Model output for {prop} is not empty for empty regions")
                pieces = [prediction.reshape(-1)[:0]] * len(samples)
            else:
                if prediction.numel() % total:
                    raise ValueError(f"Model output shape {tuple(prediction.shape)} does not match batched {prop}")
                pieces = torch.split(prediction.reshape(total, -1), counts, dim=0)
            for row, (sample, piece) in enumerate(zip(samples, pieces)):
                target = torch.as_tensor(sample[prop], device=self.config.device, dtype=self.config.dtype)
                if piece.numel() != target.numel():
                    raise ValueError(f"Model output shape {tuple(piece.shape)} does not match {prop} target {tuple(target.shape)}")
                error = piece.reshape(target.shape) - target
                if not torch.isfinite(error).all():
                    raise ValueError(f"Non-finite prediction or target for {prop}")
                errors[row][prop] = (error, target, piece.reshape(target.shape))
        if on_prediction is not None:
            # Same callback order as the per-structure path: sample-major, then target.
            for row in errors:
                for prop, (_, target, prediction) in row.items():
                    on_prediction(prop, target.detach().cpu().double().numpy(),
                                  prediction.detach().cpu().double().numpy())
        return [{prop: value[0] for prop, value in row.items()} for row in errors]

    def _losses(self, errors):
        losses = {prop: (error.square().mean() if error.numel() else error.sum())
                  for prop, error in errors.items()}
        total = sum(self.config.loss_weights[k] * v for k, v in losses.items())
        if not torch.isfinite(total):
            raise ValueError("Non-finite training loss")
        return total, losses

    def _loss(self, sample):
        return self._losses(self._errors(sample))

    def _chunk_errors(self, samples, on_prediction=None):
        """Yield per-structure error dicts in sample order (batched when possible)."""
        batched = self._batched_errors(samples, on_prediction)
        if batched is not None:
            return batched
        if on_prediction is None:
            return [self._errors(sample) for sample in samples]
        return [self._errors(sample, on_prediction=on_prediction) for sample in samples]

    @property
    def _eval_batch_size(self):
        return self.config.eval_batch_size or self.config.batch_size

    def evaluate(self, data, *, on_progress=None, should_stop=None, plot_dir=None,
                 artifact_dir=None, artifact_metadata=None):
        """Evaluate all scalar components without backward or optimizer updates.

        Energy is per structure; forces are pooled over all Cartesian components.
        Metrics use the dataset's converted units, with no energy-offset fitting.
        """
        if (plot_dir is not None or artifact_dir is not None) and not getattr(getattr(self, "backend", None), "supports_evaluation_artifacts", True):
            raise ValueError("QMMM plot/artifact export is not implemented; use numeric evaluate() metrics")
        data = data if isinstance(data, (NpyDataset, NpyShards)) else NpyDataset(data)
        self.model.eval()
        totals = {key: dict(count=0, absolute=0., squared=0.) for key in self.config.loss_weights}
        observations = ({key: [] for key in totals}
                        if plot_dir is not None or artifact_dir is not None else None)

        def collect(key, reference, prediction):
            observations[key].append((reference, prediction))
        size = self._eval_batch_size
        for start in range(0, len(data), size):
            if should_stop is not None and should_stop():
                raise TrainingStopped("Evaluation stopped by user")
            samples = [data[index] for index in range(start, min(start + size, len(data)))]
            # Forces require differentiation with respect to positions.
            with torch.enable_grad():
                chunk = self._chunk_errors(samples, on_prediction=collect if observations is not None else None)
            for errors in chunk:
                for key, error in errors.items():
                    error = error.detach().to(device="cpu", dtype=torch.float64)
                    totals[key]["count"] += error.numel()
                    totals[key]["absolute"] += error.abs().sum().item()
                    totals[key]["squared"] += error.square().sum().item()
            del chunk
            if on_progress is not None:
                on_progress({"phase": "evaluation", "completed": start + len(samples), "total": len(data)})
        metrics = {}
        for key, total in totals.items():
            if total["count"] == 0:
                metrics[key] = dict(mae=None, mse=None, rmse=None, count=0)
                continue
            mse = total["squared"] / total["count"]
            metrics[key] = dict(mae=total["absolute"] / total["count"],
                                mse=mse, rmse=math.sqrt(mse), count=total["count"])
        result = {"samples": len(data), "aggregation": "all_scalar_components",
                "units": "dataset units after configured conversion", "metrics": metrics}
        if plot_dir is not None:
            from mlpui.evaluation_plots import save_parity_plots
            if on_progress is not None:
                on_progress({"phase": "plotting", "completed": len(data), "total": len(data)})
            pairs = {key: [np.column_stack((reference.reshape(-1), prediction.reshape(-1)))
                           for reference, prediction in chunks]
                     for key, chunks in observations.items()}
            result["plots"] = save_parity_plots(pairs, metrics, plot_dir, should_stop=should_stop)
        if artifact_dir is not None:
            if should_stop is not None and should_stop():
                raise TrainingStopped("Evaluation stopped by user")
            from mlpui.evaluation_artifacts import save_evaluation_artifacts
            if on_progress is not None:
                on_progress({"phase": "saving", "completed": len(data), "total": len(data)})
            result["artifacts"] = save_evaluation_artifacts(
                observations, data, artifact_dir, metadata=artifact_metadata)
        return result

    def fit(self, train_data, validation_data=None, *, test_data=None, output_dir=None,
            on_progress=None, should_stop=None):
        def check_stop():
            if should_stop is not None and should_stop():
                raise TrainingStopped("Training stopped by user")

        check_stop()
        self.early_stopping = None
        if self.config.early_stopping:
            if validation_data is None:
                raise ValueError("Early stopping requires a validation dataset")
            if output_dir is None:
                raise ValueError("Early stopping requires output_dir to save best.pt")
            self.early_stopping = dict(monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience, min_delta=self.config.early_stopping_min_delta,
                best_value=None, best_epoch=None, reference_value=None, bad_epochs=0, stopped=False)
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
                check_stop()
                samples = [train[int(index)] for index in indices]
                with torch.enable_grad():
                    batched = self._batched_errors(samples)
                if batched is not None:
                    with torch.enable_grad():
                        results = [self._losses(errors) for errors in batched]
                        (sum(total for total, _ in results) / len(indices)).backward()
                    for _, parts in results:
                        for key, value in parts.items():
                            sums[key] += value.detach().item()
                    del batched, results
                else:
                    for sample in samples:
                        check_stop()
                        with torch.enable_grad():
                            loss, parts = self._loss(sample)
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
            evaluations = []
            if (row["epoch"] % self.config.validation_interval == 0
                    or row["epoch"] == self.config.epochs):
                evaluations.append(("validation", validation))
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
                size = self._eval_batch_size
                with torch.enable_grad():
                    for start in range(0, len(evaluation), size):
                        check_stop()
                        samples = [evaluation[index] for index in range(start, min(start + size, len(evaluation)))]
                        batched = self._batched_errors(samples)
                        results = ([self._losses(errors) for errors in batched] if batched is not None
                                   else [self._loss(sample) for sample in samples])
                        for _, parts in results:
                            for key, value in parts.items():
                                sums[key] += value.detach().item()
                        del batched, results
                row[split] = {k: v / len(evaluation) for k, v in sums.items()}
            self.history.append(row)
            if self.early_stopping is not None and "validation" in row:
                early = self.early_stopping
                values = row["validation"]
                metric = (sum(self.config.loss_weights[k] * v for k, v in values.items())
                          if early["monitor"] == "loss" else values[early["monitor"]])
                if not math.isfinite(metric):
                    raise ValueError("Non-finite early stopping metric")
                # Patience uses significant improvements; best.pt always tracks
                # the actual lowest validation value, including smaller changes.
                if early["reference_value"] is None or metric < early["reference_value"] - early["min_delta"]:
                    early.update(reference_value=metric, bad_epochs=0)
                else:
                    early["bad_epochs"] += 1
                early["stopped"] = early["bad_epochs"] >= early["patience"]
                if early["best_value"] is None or metric < early["best_value"]:
                    early.update(best_value=metric, best_epoch=row["epoch"])
                    self.save(Path(output_dir) / "best.pt")
            if self.config.save_interval and row["epoch"] % self.config.save_interval == 0:
                self._save_periodic(output_dir, row["epoch"])
            if on_progress is not None:
                on_progress({"phase": "epoch_end", "epoch": len(self.history),
                             "completed": len(train), "total": len(train),
                             "history": self.history, "early_stopping": copy.deepcopy(self.early_stopping)})
            if self.early_stopping is not None and self.early_stopping["stopped"]:
                break
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
        """Export model weights/config, not external baseline data or optimizer resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            torch.save({"backend": self.family, "format_version": 1,
                    "model_config_version": self.backend.config_version,
                    "model_config": self.model_config, "early_stopping": getattr(self, "early_stopping", None),
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
