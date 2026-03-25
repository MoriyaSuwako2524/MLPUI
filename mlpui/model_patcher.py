
from __future__ import annotations

import copy
import logging
from enum import Enum, auto
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PatchType(Enum):
    FULL = auto()
    DELTA = auto()
    LORA = auto()


class WeightPatch:

    __slots__ = ("value", "patch_type", "strength")

    def __init__(self, value: torch.Tensor, patch_type: PatchType, strength: float = 1.0):
        self.value = value
        self.patch_type = patch_type
        self.strength = strength

    def apply(self, original: torch.Tensor) -> torch.Tensor:
        if self.patch_type == PatchType.FULL:
            return self.value.to(dtype=original.dtype, device=original.device)
        elif self.patch_type == PatchType.DELTA:
            delta = self.value.to(dtype=original.dtype, device=original.device)
            return original + self.strength * delta
        else:
            raise NotImplementedError(f"PatchType {self.patch_type} not yet supported")



def _get_attr(obj: nn.Module, key: str) -> torch.Tensor:

    parts = key.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    leaf = parts[-1]

    param = getattr(obj, leaf)
    if isinstance(param, nn.Parameter):
        return param.data
    return param


def _set_attr(obj: nn.Module, key: str, value: torch.Tensor) -> None:

    parts = key.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    leaf = parts[-1]
    current = getattr(obj, leaf)
    if isinstance(current, nn.Parameter):
        current.data.copy_(value)
    else:
        setattr(obj, leaf, value)




class ModelPatcher:

    def __init__(
        self,
        model: nn.Module,
        load_device: torch.device | None = None,
        offload_device: torch.device | None = None,
    ):
        self.model = model
        self.load_device = load_device or torch.device("cpu")
        self.offload_device = offload_device or torch.device("cpu")

        # key → list[WeightPatch]  (multiple patches can stack on one key)
        self.patches: dict[str, list[WeightPatch]] = {}

        # key → original Tensor (populated by patch_model, cleared by unpatch_model)
        self._backup: dict[str, torch.Tensor] = {}

        # Whether patches are currently applied to self.model
        self._is_patched: bool = False



    def add_patch(
        self,
        key: str,
        value: torch.Tensor,
        patch_type: PatchType = PatchType.DELTA,
        strength: float = 1.0,
    ) -> None:
        """
        Register a weight patch for a single parameter.

        Args:
            key: Dot-separated parameter path (e.g. 'blocks.0.edge_wise.weight').
            value: The patch tensor (replacement for FULL, delta for DELTA).
            patch_type: How the patch modifies the original weight.
            strength: Scaling factor (only used for DELTA/LORA).
        """
        if key not in self.patches:
            self.patches[key] = []
        self.patches[key].append(WeightPatch(value, patch_type, strength))
        logger.debug(f"Registered {patch_type.name} patch for '{key}' (strength={strength})")

    def add_patches_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        patch_type: PatchType = PatchType.FULL,
        strength: float = 1.0,
        prefix: str = "",
    ) -> tuple[list[str], list[str]]:
        """
        Register patches for all keys in a state_dict.

        Args:
            state_dict: Mapping of parameter paths to tensors.
            patch_type: How each patch modifies the original weight.
            strength: Scaling factor applied to all patches.
            prefix: Prefix to strip from state_dict keys before matching.

        Returns:
            (matched_keys, unmatched_keys) — keys that did/didn't
            correspond to existing model parameters.
        """
        model_keys = set(dict(self.model.named_parameters()).keys())
        model_keys |= set(dict(self.model.named_buffers()).keys())

        matched, unmatched = [], []
        for raw_key, tensor in state_dict.items():
            key = raw_key[len(prefix):] if prefix and raw_key.startswith(prefix) else raw_key
            if key in model_keys:
                self.add_patch(key, tensor, patch_type, strength)
                matched.append(key)
            else:
                unmatched.append(raw_key)

        logger.info(
            f"Registered {len(matched)} {patch_type.name} patches "
            f"(strength={strength}), {len(unmatched)} unmatched keys"
        )
        return matched, unmatched

    def clear_patches(self) -> None:
        """Remove all registered patches (does NOT unpatch if currently applied)."""
        self.patches.clear()


    def _calculate_patched_weight(self, key: str, original: torch.Tensor) -> torch.Tensor:

        result = original.clone()

        for patch in self.patches.get(key, []):
            if patch.patch_type == PatchType.FULL:
                # Full replacement — discard everything before this
                result = patch.value.to(dtype=original.dtype, device=original.device)
            else:
                result = patch.apply(result)

        return result

    def patch_model(self, device: torch.device | None = None) -> nn.Module:
        """
        Backup original weights and write patched weights into the backbone.

        Call unpatch_model() when done to restore originals.

        Args:
            device: Optionally move model to this device before patching.

        Returns:
            The patched backbone module (same object as self.model).
        """
        if self._is_patched:
            logger.warning("patch_model() called but patches already applied — unpatching first")
            self.unpatch_model()

        if device is not None:
            self.model.to(device)

        for key in self.patches:
            try:
                original = _get_attr(self.model, key)
            except (AttributeError, IndexError) as e:
                logger.warning(f"Skipping patch for '{key}': {e}")
                continue

            # Backup
            self._backup[key] = original.clone()

            # Calculate and write patched weight
            patched = self._calculate_patched_weight(key, original)
            _set_attr(self.model, key, patched)

        self._is_patched = True
        logger.info(f"Applied patches to {len(self._backup)} parameters")
        return self.model

    def unpatch_model(self, device: torch.device | None = None) -> nn.Module:
        """
        Restore all backed-up original weights.

        Args:
            device: Optionally move model to this device after restoring.

        Returns:
            The restored backbone module.
        """
        if not self._is_patched:
            logger.debug("unpatch_model() called but no patches applied — no-op")
            return self.model

        for key, original in self._backup.items():
            try:
                _set_attr(self.model, key, original)
            except (AttributeError, IndexError) as e:
                logger.warning(f"Failed to restore '{key}': {e}")

        self._backup.clear()
        self._is_patched = False

        if device is not None:
            self.model.to(device)

        logger.info("Restored original weights")
        return self.model


    def clone(self) -> ModelPatcher:
        """
        Create a lightweight copy that shares the backbone but has
        independent patches.

        This is the key mechanism for multi-property prediction from a
        single foundation model:

            p_energy = patcher.clone()
            p_dipole = patcher.clone()
            p_energy.add_patches_from_state_dict(energy_finetune_sd)
            p_dipole.add_patches_from_state_dict(dipole_finetune_sd)

        Both share the same backbone memory. patch_model()/unpatch_model()
        must be used sequentially (not concurrently) since they mutate
        the shared backbone.
        """
        new = ModelPatcher(
            model=self.model,  # shared reference
            load_device=self.load_device,
            offload_device=self.offload_device,
        )
        # Deep copy the patch lists so modifications are independent
        new.patches = {k: list(v) for k, v in self.patches.items()}
        return new

    # ------------------------------------------------------------------


    def to_device(self, device: torch.device | None = None) -> None:
        """Move backbone to the specified device (default: load_device)."""
        device = device or self.load_device
        self.model.to(device)

    def offload(self) -> None:
        """Move backbone to the offload device (typically CPU) to free GPU memory."""
        self.model.to(self.offload_device)
        if self.offload_device.type != "cpu":
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def model_size(self) -> int:
        """Estimate model memory footprint in bytes."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())



    def load_finetune(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str = "",
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Shorthand for registering a full fine-tune as FULL patches.

        Equivalent to:
            self.add_patches_from_state_dict(sd, PatchType.FULL, prefix=prefix)

        Returns:
            (matched_keys, unmatched_keys)
        """
        return self.add_patches_from_state_dict(
            state_dict, PatchType.FULL, strength=1.0, prefix=prefix
        )

    def load_delta(
        self,
        state_dict: dict[str, torch.Tensor],
        strength: float = 1.0,
        prefix: str = "",
    ) -> tuple[list[str], list[str]]:
        """
        Shorthand for registering a delta fine-tune (W_new = W_base + strength * delta).

        The state_dict should contain weight DIFFERENCES, not absolute weights.
        To create deltas from two checkpoints:
            delta_sd = {k: ft_sd[k] - base_sd[k] for k in ft_sd}

        Returns:
            (matched_keys, unmatched_keys)
        """
        return self.add_patches_from_state_dict(
            state_dict, PatchType.DELTA, strength=strength, prefix=prefix
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_dtype(self) -> torch.dtype:
        """Infer the model's primary dtype from its parameters."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def is_patched(self) -> bool:
        return self._is_patched

    @property
    def num_patches(self) -> int:
        return sum(len(v) for v in self.patches.values())
    def is_dynamic(self):
        return False

    def __repr__(self) -> str:
        cls = type(self.model).__name__
        n = self.num_patches
        patched = " [PATCHED]" if self._is_patched else ""
        return f"ModelPatcher({cls}, patches={n}, device={self.load_device}{patched})"