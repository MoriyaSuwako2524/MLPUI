"""Checkpoint adapters for the bundled TorchMD-Net and NewtonNet models.

The bundled implementations are imported only when their family is selected.
All weights are checked strictly, including learned output heads and scalers.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import pickle
from pathlib import Path
import re

import torch
from torch import nn

from mlpui.model_patcher import ModelPatcher


def _config_dict(config):
    if config is None:
        return {}
    if isinstance(config, (str, Path)):
        import yaml
        with open(config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("model_config must be a dictionary or a YAML path")
    return copy.deepcopy(dict(config))


def _strip_prefix(state):
    state = dict(state)
    for prefix in ("module.", "model."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def detect_family(state, config=None):
    """Identify a family without importing either optional backend."""
    if isinstance(state, nn.Module):
        name = type(state).__module__
        if name.startswith(("newtonnet.", "mlpui.models.newtonnet.")):
            return "newtonnet"
        if name.startswith(("torchmdnet.", "mlpui.models.torchmdnet.")):
            return "torchmdnet"
        return None
    if not isinstance(state, Mapping):
        return None
    keys = _strip_prefix(state)
    if any(k.startswith("representation_model.") for k in keys):
        return "torchmdnet"
    if "embedding_layers.node_embedding.weight" in keys:
        return "newtonnet"
    return None


def _torchmd_model(state, config):
    try:
        from mlpui.models.torchmdnet.models.model import create_model, TorchMD_Net
    except ImportError as exc:
        raise ImportError(
            "Bundled TorchMD-Net dependencies are unavailable; see README.md."
        ) from exc
    if not config or "model" not in config:
        raise ValueError("TorchMD-Net needs checkpoint hyper_parameters or model_config")
    model = create_model(copy.deepcopy(config))
    # The fork creates a multi-output model even for legacy single-head checkpoints.
    if any(k.startswith("output_model.") for k in state) and hasattr(model, "output_modules"):
        model = TorchMD_Net(
            model.representation_model, model.output_modules["y"],
            prior_model=model.prior_model, derivative=config.get("derivative", False),
            dtype=next(model.parameters()).dtype,
        )
    patterns = [
        (r"output_model\.output_network\.(\d+)\.update_net\.(\d+)\.",
         r"output_model.output_network.\1.update_net.layers.\2."),
        (r"output_model\.output_network\.([02])\.(weight|bias)",
         r"output_model.output_network.layers.\1.\2"),
    ]
    for old, new in patterns:
        state = {re.sub(old, new, k): v for k, v in state.items()}
    # Restore per-head normalizations without the fork's non-strict override.
    if hasattr(model, "output_modules"):
        for prefix in ("mean", "std"):
            for head in model.output_modules:
                key = f"{prefix}_{head}"
                if key in state:
                    model.register_buffer(key, torch.empty_like(state[key]))
            if prefix not in state and all(f"{prefix}_{h}" in state for h in model.output_modules):
                if prefix in model._buffers:
                    delattr(model, prefix)
    nn.Module.load_state_dict(model, state, strict=True)
    model.mlpui_output_model = config.get("output_model", "Scalar")
    return model


def _newton_model(state, config):
    try:
        from mlpui.models.newtonnet.models.newtonnet import NewtonNet
    except ImportError as exc:
        raise ImportError(
            "Bundled NewtonNet dependencies are unavailable; see README.md."
        ) from exc
    if isinstance(config.get("model"), Mapping):
        config = config["model"]
    if not config or "output_properties" not in config:
        raise ValueError(
            "NewtonNet state_dict needs model_config with output_properties, "
            "cutoff and architecture settings (or the training config.yml). "
            "These cannot be inferred reliably from tensor shapes."
        )
    config = dict(config)
    config.pop("pretrained_model", None)  # training-only initialization directive
    model = NewtonNet(**config)
    # Preserve double-precision checkpoints and learned scale/shift parameters.
    dtype = next((v.dtype for v in state.values() if v.is_floating_point()), torch.float32)
    model.to(dtype=dtype)
    nn.Module.load_state_dict(model, state, strict=True)
    return model


def load_external_checkpoint(path, *, family=None, model_config=None, device=None,
                             dtype=None, trusted_checkpoint=False):
    """Return a patcher, or None for a checkpoint belonging to the UMA loader.

    Serialized nn.Module files require explicit trusted_checkpoint=True because
    pickle can execute code. State-dict checkpoints use weights_only=True.
    """
    aliases = {"torchmd-net": "torchmdnet", "tensornet": "torchmdnet"}
    family = aliases.get(family, family)
    if family not in (None, "uma", "torchmdnet", "newtonnet"):
        raise ValueError(f"Unknown model family: {family}")
    if family == "uma":
        return None
    try:
        options = {}
        if trusted_checkpoint:
            from mlpui.models import checkpoint_pickle
            options["pickle_module"] = checkpoint_pickle
        checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted_checkpoint, **options)
    except pickle.UnpicklingError as exc:
        if family is None and "fairchem.core.units.mlip_unit.api.inference" in str(exc):
            return None
        raise ValueError(
            "This checkpoint is not a weights-only file. For a trusted, serialized "
            "model or Lightning checkpoint use trusted_checkpoint=True; "
            "otherwise export a state_dict with model_config."
        ) from exc
    except AttributeError as exc:
        raise ValueError(
            "Checkpoint references classes absent from the installed backend. "
            "Export a state_dict and matching model_config in its original "
            "training environment, or use the matching backend revision."
        ) from exc
    config = {}
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        config = checkpoint.get("hyper_parameters") or checkpoint.get("model_config") or checkpoint.get("config") or {}
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    saved_config = _config_dict(config)
    config = _config_dict(model_config if model_config is not None else config)
    saved_settings = saved_config.get("model", saved_config)
    if not isinstance(saved_settings, Mapping):
        saved_settings = saved_config
    settings = config.get("model", config)
    if not isinstance(settings, Mapping):
        settings = config
    constrained = saved_settings.get("charge_constraint", False) or getattr(state, "charge_constraint", False)
    if constrained:
        if settings.get("charge_constraint") is False:
            raise ValueError("Checkpoint enables total-charge constraint; it cannot be silently disabled")
        settings["charge_constraint"] = True
    detected = detect_family(state, config)
    if family is not None and detected is not None and family != detected:
        raise ValueError(f"Checkpoint is {detected}, not requested family {family}")
    family = family or detected
    if family is None:
        return None
    if isinstance(state, nn.Module):
        if detected != family:
            raise ValueError("Serialized model does not belong to the requested backend")
        model = state
    else:
        if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected a tensor state_dict")
        state = _strip_prefix(state)
        model = _torchmd_model(state, config) if family == "torchmdnet" else _newton_model(state, config)
    model.mlpui_family = family
    if settings.get("charge_constraint", False):
        heads = getattr(model, "output_properties", []) if family == "newtonnet" else getattr(model, "output_modules", {})
        if "charge" not in heads:
            raise ValueError("Total-charge constraint requires a charge output head")
        model.charge_constraint = True
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return ModelPatcher(model, load_device=device, offload_device=torch.device("cpu"))
