"""Small, explicit configuration contract for the basic web interface."""
from pathlib import Path
import copy

from mlpui.data import NpyDataset, NpyShards


def presets():
    return {
        "newtonnet": dict(cutoff=5., n_features=64, n_basis=20, n_interactions=3,
                          activation="silu", output_properties=["energy", "gradient_force"]),
        "torchmdnet": dict(model="tensornet", precision=32, embedding_dimension=64,
            num_layers=3, num_rbf=32, rbf_type="expnorm", trainable_rbf=False,
            activation="silu", cutoff_lower=0., cutoff_upper=5., max_z=119,
            max_num_neighbors=64, aggr="add", neighbor_embedding=True,
            attn_activation="silu", num_heads=4, distance_influence="both",
            equivariance_invariance_group="O(3)", derivative=True, atom_filter=-1,
            prior_model=None, output_model="Scalar", reduce_op="sum",
            pred_dict={"y": 1., "neg_dy": 1.}),
    }


def dataset(spec):
    options = copy.deepcopy(spec)
    directory = options.pop("directory")
    return (NpyShards if "shards" in options else NpyDataset)(directory, **options)


def normalize(payload):
    from mlpui.training import TrainingConfig, configure_charge_head
    import torch
    value = copy.deepcopy(payload)
    if not isinstance(value, dict):
        raise ValueError("Expected a configuration object")
    allowed = {"name", "family", "model_config", "training", "train", "validation", "test", "checkpoint", "task_type", "evaluation"}
    if value.keys() - allowed:
        raise ValueError("Unsupported configuration fields")
    value.setdefault("task_type", "training")
    if value["task_type"] not in ("training", "evaluation"):
        raise ValueError("Task type must be training or evaluation")
    evaluating = value["task_type"] == "evaluation"
    if evaluating and any(key in value for key in ("train", "validation", "test")):
        raise ValueError("Evaluation tasks use only the evaluation dataset")
    if not evaluating and "evaluation" in value:
        raise ValueError("Training tasks use train, validation and test datasets")
    value["name"] = str(value.get("name", "Training run")).strip()
    if not value["name"] or len(value["name"]) > 100:
        raise ValueError("Task name must contain 1–100 characters")
    if value.get("family") not in presets():
        raise ValueError("Select NewtonNet or TorchMD-Net")
    if not isinstance(value.get("model_config"), dict) or not value["model_config"]:
        raise ValueError("Model configuration must be a nonempty JSON object")
    options = dict(value.get("training", {}))
    dtype = options.get("dtype", "float32")
    if dtype not in ("float32", "float64"):
        raise ValueError("Precision must be float32 or float64")
    TrainingConfig(**{**options, "dtype": getattr(torch, dtype)})
    value["training"] = options
    value["model_config"] = configure_charge_head(
        value["family"], value["model_config"], options.get("loss_weights", {}))
    for key in (("evaluation",) if evaluating else ("train", "validation", "test")):
        spec = value.get(key)
        if key in ("validation", "test") and not spec:
            value.pop(key, None)
            continue
        if not isinstance(spec, dict) or not spec.get("directory"):
            raise ValueError("A dataset directory is required")
        if spec.keys() - {"directory", "shards", "files", "gradients", "energy_scale", "length_scale"}:
            raise ValueError("Unsupported dataset fields")
        spec["directory"] = str(Path(spec["directory"]).expanduser().resolve())
        if "files" in spec and (not isinstance(spec["files"], dict) or
                                not all(isinstance(k, str) and isinstance(v, str) and v
                                        for k, v in spec["files"].items())):
            raise ValueError("File mapping must map field names to .npy filenames")
        if "gradients" in spec and not isinstance(spec["gradients"], bool):
            raise ValueError("gradients must be true or false")
        if "shards" in spec and (not isinstance(spec["shards"], list) or
                                 not all(isinstance(s, str) and s for s in spec["shards"])):
            raise ValueError("Group names must be a list of nonempty strings")
    if value.get("checkpoint"):
        checkpoint = Path(value["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError("Checkpoint file does not exist")
        value["checkpoint"] = str(checkpoint)
    else:
        value.pop("checkpoint", None)
    if evaluating and not value.get("checkpoint"):
        raise ValueError("Evaluation requires an existing model checkpoint")
    def coordinate_files(spec):
        template = spec.get("files", {}).get("pos", "pos.npy")
        return {(Path(spec["directory"]) / template.format(shard=s)).resolve()
                for s in spec.get("shards", [""])}

    splits = [key for key in ("train", "validation", "test") if key in value]
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            if coordinate_files(value[first]) & coordinate_files(value[second]):
                raise ValueError(f"{first} and {second} data must be separate")
    if options.get("early_stopping", False) and (evaluating or not value.get("validation")):
        raise ValueError("早停只适用于训练，并且必须提供独立验证集")
    return value


def inspect_data(settings):
    result = {}
    labels = set(settings["training"].get("loss_weights", {"energy": 1, "forces": 1}))
    model_config = settings["model_config"]
    if settings["family"] == "newtonnet":
        model_config = model_config.get("model", model_config)
    if model_config.get("charge_constraint", False):
        labels.add("charge")
    for key in ("train", "validation", "test", "evaluation"):
        if key not in settings:
            continue
        data = dataset(settings[key])
        fields = data.fields if isinstance(data, NpyShards) else data.arrays.keys()
        missing = labels - fields
        if missing:
            raise ValueError(f"{key} missing labels: {', '.join(sorted(missing))}")
        parts = data.datasets if isinstance(data, NpyShards) else [data]
        result[key] = {"samples": len(data), "groups": len(parts),
                       "fields": sorted(fields), "first_atoms": len(data[0]["z"])}
    return result
