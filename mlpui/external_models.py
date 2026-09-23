"""Checkpoint dispatch for registered model backends.

The bundled implementations are imported only when their family is selected.
All weights are checked strictly, including learned output heads and scalers.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import pickle
from pathlib import Path

import torch
from torch import nn

from mlpui.model_patcher import ModelPatcher
from mlpui.backends import get_backend, find_backend


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
    """Identify registered backends without importing their model packages."""
    if not isinstance(state, (nn.Module, Mapping)):
        return None
    backend = find_backend(_strip_prefix(state) if isinstance(state, Mapping) else state)
    return backend.name if backend else None


def load_external_checkpoint(path, *, family=None, model_config=None, device=None,
                             dtype=None, trusted_checkpoint=False):
    """Return a patcher, or None for a checkpoint belonging to the UMA loader.

    Serialized nn.Module files require explicit trusted_checkpoint=True because
    pickle can execute code. State-dict checkpoints use weights_only=True.
    """
    if family == "uma":
        return None
    if family is not None:
        family = get_backend(family).name
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
        if "format_version" in checkpoint:
            if type(checkpoint["format_version"]) is not int or checkpoint["format_version"] != 1:
                raise ValueError("Unsupported checkpoint format_version")
            if "backend" not in checkpoint:
                raise ValueError("Versioned checkpoint requires backend metadata")
        if "backend" in checkpoint:
            saved_backend = get_backend(checkpoint["backend"])
            if family is not None and family != saved_backend.name:
                raise ValueError(f"Checkpoint is {saved_backend.name}, not requested family {family}")
            family = saved_backend.name
            version = checkpoint.get("model_config_version", 1)
            if type(version) is not int or version != saved_backend.config_version:
                raise ValueError("Unsupported checkpoint model_config_version")
        config = checkpoint.get("hyper_parameters") or checkpoint.get("model_config") or checkpoint.get("config") or {}
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    saved_config = _config_dict(config)
    config = _config_dict(model_config if model_config is not None else config)
    detected = detect_family(state, config)
    if family is not None and detected is not None and family != detected:
        raise ValueError(f"Checkpoint is {detected}, not requested family {family}")
    family = family or detected
    if family is None:
        return None
    backend = get_backend(family)
    saved_settings = backend.settings(saved_config)
    settings = backend.settings(config)
    constrained = saved_settings.get("charge_constraint", False) or getattr(state, "charge_constraint", False)
    if constrained:
        if settings.get("charge_constraint") is False:
            raise ValueError("Checkpoint enables total-charge constraint; it cannot be silently disabled")
        settings["charge_constraint"] = True
    if isinstance(state, nn.Module):
        if detected != family:
            raise ValueError("Serialized model does not belong to the requested backend")
        model = state
    else:
        if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected a tensor state_dict")
        state = _strip_prefix(state)
        model = backend.load_model(state, config)
    model.mlpui_family = family
    if settings.get("charge_constraint", False):
        backend.validate_constraint(model)
        model.charge_constraint = True
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return ModelPatcher(model, load_device=device, offload_device=torch.device("cpu"))
