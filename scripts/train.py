"""Run `python -m scripts.train path/to/training.yaml`."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from mlpui.data import NpyDataset, NpyShards
from mlpui.training import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    base = args.config.resolve().parent

    def path(value):
        return str(base / value)

    def dataset(spec):
        spec = dict(spec)
        directory = path(spec.pop("directory"))
        return (NpyShards if "shards" in spec else NpyDataset)(directory, **spec)

    options = dict(settings.get("training", {}))
    dtype = options.pop("dtype", "float32")
    if dtype not in ("float32", "float64"):
        parser.error("dtype must be float32 or float64")
    options["dtype"] = getattr(torch, dtype)
    model = settings["model_config"]
    trainer = Trainer(settings["family"], path(model) if isinstance(model, str) else model,
                      TrainingConfig(**options),
                      checkpoint=path(settings["checkpoint"]) if settings.get("checkpoint") else None,
                      trusted_checkpoint=settings.get("trusted_checkpoint", False))
    history = trainer.fit(dataset(settings["train"]),
                          dataset(settings["validation"]) if "validation" in settings else None,
                          output_dir=path(settings.get("output_dir", "run")))
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
