
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import ase
import ase.calculators.calculator as ase_calc
from ase.calculators.calculator import Calculator, all_changes

logger = logging.getLogger(__name__)



class InputAdapter(ABC):

    @abstractmethod
    def convert(self, atoms: ase.Atoms, device: torch.device, dtype: torch.dtype) -> Any:
        """Convert ASE Atoms to model-ready input."""
        ...


class OutputAdapter(ABC):


    @abstractmethod
    def convert(self, output: Any, atoms: ase.Atoms) -> dict[str, np.ndarray]:
        """Convert model output to ASE results dict."""
        ...




class UMAInputAdapter(InputAdapter):


    def __init__(self, task: str | None = None):
        self.task = task

    def convert(self, atoms: ase.Atoms, device: torch.device, dtype: torch.dtype) -> Any:
        from mlpui.models.common.atomic_data import AtomicData

        # Ensure defaults so atoms without info["charge"/"spin"] work correctly.
        # from_ase reads these via r_data_keys — no manual override needed.
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin",   0)

        data = AtomicData.from_ase(
            atoms,
            task_name=self.task,
            r_edges=False,
            r_data_keys=["charge", "spin"],
        )


        data["natoms"] = torch.tensor([len(atoms)], dtype=torch.long)
        if self.task is not None:
            data["dataset"] = [self.task]

        if "batch" not in data or data.get("batch", None) is None:
            data["batch"] = torch.zeros(len(atoms), dtype=torch.long)

        return data.to(device)


class UMAOutputAdapter(OutputAdapter):
    """Output adapter for UMA-family models."""

    def __init__(self, properties: list[str]):
        self.properties = properties

    def convert(self, output: dict, atoms: ase.Atoms) -> dict[str, np.ndarray]:
        results = {}
        n = len(atoms)

        if "energy" in self.properties:
            for key in ("energy", "total_energy", "y"):
                if key in output:
                    results["energy"] = float(output[key].detach().cpu().squeeze())
                    break

        if "forces" in self.properties:
            for key in ("forces", "neg_dy", "force"):
                if key in output:
                    results["forces"] = output[key].detach().cpu().numpy().reshape(n, 3)
                    break


        return results



class SimpleInputAdapter(InputAdapter):
    """Input adapter for TorchMD-Net style models (z, pos, batch)."""

    def __init__(self, needs_grad: bool = True):
        self.needs_grad = needs_grad

    def convert(self, atoms: ase.Atoms, device: torch.device, dtype: torch.dtype) -> dict:
        n = len(atoms)
        pos = torch.tensor(atoms.positions, dtype=dtype, device=device)
        if self.needs_grad:
            pos = pos.requires_grad_(True)

        return {
            "z": torch.tensor(atoms.numbers, dtype=torch.long, device=device),
            "pos": pos,
            "batch": torch.zeros(n, dtype=torch.long, device=device),
        }


class SimpleOutputAdapter(OutputAdapter):
    """Output adapter for models returning (energy, forces) tuple or dict."""

    def __init__(self, properties: list[str]):
        self.properties = properties

    def convert(self, output: Any, atoms: ase.Atoms) -> dict[str, np.ndarray]:
        if isinstance(output, tuple):
            output = {"energy": output[0]}
            if len(output) > 1 and output[1] is not None:
                output["forces"] = output[1]

        results = {}
        n = len(atoms)

        if "energy" in self.properties:
            for key in ("energy", "y"):
                if key in output:
                    results["energy"] = float(output[key].detach().cpu().squeeze())
                    break

        if "forces" in self.properties:
            for key in ("forces", "neg_dy"):
                if key in output:
                    results["forces"] = output[key].detach().cpu().numpy().reshape(n, 3)
                    break

        if "dipole" in self.properties:
            for key in ("dipole", "vec"):
                if key in output:
                    results["dipole"] = output[key].detach().cpu().numpy().reshape(3)
                    break

        return results




class MLPCalculator(Calculator):

    implemented_properties = ["energy", "forces", "stress", "dipole"]

    def __init__(
        self,
        model_patcher,
        input_adapter: InputAdapter,
        output_adapter: OutputAdapter,
        properties: list[str],
        dtype: torch.dtype,
        device: torch.device,
        keep_on_device: bool,
    ):
        super().__init__()
        self._patcher = model_patcher
        self._input_adapter = input_adapter
        self._output_adapter = output_adapter
        self._properties = properties
        self._dtype = dtype
        self._device = device
        self._keep_on_device = keep_on_device
        self._ready = False
        self._last_charge: int | None = None
        self._last_spin: int | None = None

    def check_state(self, atoms: ase.Atoms, tol: float = 1e-15) -> list[str]:
        # ASE's default check_state only compares geometry, not atoms.info.
        # Extend it to invalidate cache when charge or spin changes.
        changes = super().check_state(atoms, tol)
        charge = atoms.info.get("charge", 0)
        spin   = atoms.info.get("spin",   0)
        if charge != self._last_charge or spin != self._last_spin:
            if "charge_spin" not in changes:
                changes.append("charge_spin")
        return changes



    def _ensure_ready(self):

        if self._ready:
            return

        self._patcher.to_device(self._device)
        if self._patcher.num_patches > 0:
            self._patcher.patch_model(device=self._device)

        backbone = self._get_backbone()
        backbone.eval()

        self._ready = True

    def _maybe_offload(self):
        """
        Unpatch and optionally offload after inference.
        ≈ ComfyUI unpatch_model()
        """
        if self._patcher.is_patched:
            self._patcher.unpatch_model()
        if not self._keep_on_device:
            self._patcher.offload()
            self._ready = False

    def _get_backbone(self) -> nn.Module:
        model = self._patcher.model
        return model.diffusion_model if hasattr(model, "diffusion_model") else model



    def _forward(self, model_input) -> Any:

        model = self._patcher.model
        backbone = self._get_backbone()

        if hasattr(model, "apply_model"):
            return model.apply_model(model_input)

        if hasattr(model_input, "atomic_numbers"):
            needs_grad = "forces" in self._properties
            if needs_grad and hasattr(model_input, "pos") and not model_input.pos.requires_grad:
                model_input.pos.requires_grad_(True)
            ctx = torch.enable_grad() if needs_grad else torch.no_grad()
            with ctx:
                return backbone(model_input)

        pos = model_input["pos"]
        ctx = torch.enable_grad() if pos.requires_grad else torch.no_grad()
        with ctx:
            return backbone(model_input["z"], pos, model_input["batch"])

    def _check_atoms_pbc(self, atoms) -> None:

        if np.all(atoms.pbc) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise MixedPBCError

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):

        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms

        try:
            self._ensure_ready()
            model_input = self._input_adapter.convert(atoms, self._device, self._dtype)
            with torch.no_grad() if "forces" not in self._properties else torch.enable_grad():
                raw_output = self._forward(model_input)
            self.results = self._output_adapter.convert(raw_output, atoms)
            self._last_charge = atoms.info.get("charge", 0)
            self._last_spin   = atoms.info.get("spin",   0)
        finally:
            self._maybe_offload()


    def __enter__(self):

        self._saved_keep = self._keep_on_device
        self._keep_on_device = True
        self._ensure_ready()
        return self

    def __exit__(self, *args):
        self._keep_on_device = self._saved_keep
        self._maybe_offload()

    def __repr__(self):
        bb = type(self._get_backbone()).__name__
        return f"MLPCalculator({bb}, properties={self._properties}, device={self._device})"




class UMACalculator(MLPCalculator):
    def __init__(self,model_patcher,input_adapter: InputAdapter,output_adapter: OutputAdapter,
                 properties: list[str],dtype: torch.dtype,device: torch.device,keep_on_device: bool,
    ):
        super().__init__(model_patcher,input_adapter,output_adapter,properties,dtype,device,keep_on_device)

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        self._check_atoms_pbc(atoms)



        super().calculate(atoms, properties, system_changes)
    def _forward(self, model_input) -> Any:
        model = self._patcher.model
        backbone = self._get_backbone()
        if not getattr(self, '_inference_prepared', False):
            if hasattr(backbone, 'prepare_for_inference'):
                from mlpui.models.uma.inference import InferenceSettings
                settings = InferenceSettings()
                new_backbone = backbone.prepare_for_inference(model_input, settings)
                if new_backbone is not backbone:
                    model.diffusion_model = new_backbone
                    backbone = new_backbone
            self._inference_prepared = True

        if hasattr(model, "apply_model"):
            return model.apply_model(model_input)

        if hasattr(model_input, "atomic_numbers"):
            needs_grad = "forces" in self._properties
            if needs_grad and hasattr(model_input, "pos") and not model_input.pos.requires_grad:
                model_input.pos.requires_grad_(True)
            ctx = torch.enable_grad() if needs_grad else torch.no_grad()
            with ctx:
                backbone_out = backbone(model_input)
                if hasattr(model, "output_head") and model.output_head is not None:
                    dataset = None
                    try:
                        d = model_input["dataset"]
                        dataset = d[0] if isinstance(d, (list, tuple)) else d
                    except (KeyError, AttributeError, TypeError):
                        pass
                    return model.output_head(
                        backbone_out,
                        model_input["pos"],
                        model_input["atomic_numbers"],
                        dataset,
                    )
                return backbone_out

        pos = model_input["pos"]
        ctx = torch.enable_grad() if pos.requires_grad else torch.no_grad()
        with ctx:
            return backbone(model_input["z"], pos, model_input["batch"])
    def _check_atoms_pbc(self, atoms) -> None:
        if np.all(atoms.pbc) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise MixedPBCError
    def _ensure_ready(self):

        if self._ready:
            return

        self._patcher.to_device(self._device)
        if self._patcher.num_patches > 0:
            self._patcher.patch_model(device=self._device)

        backbone = self._get_backbone()
        backbone.eval()
        backbone.always_use_pbc = False
        backbone.otf_graph = True
        self._ready = True

@dataclass
class CalculatorBuilder:


    model_patcher: Any
    properties: list[str] = field(default_factory=lambda: ["energy", "forces"])
    task: str | None = None
    dtype: torch.dtype = torch.float32
    device: torch.device | str | None = None
    keep_on_device: bool = True

    def build(self):

        device = self._resolve_device()
        input_adapter = self._resolve_input_adapter()
        output_adapter = self._resolve_output_adapter()

        logger.info(
            "Built calculator: backbone=%s, input=%s, output=%s, device=%s",
            type(self._get_backbone()).__name__,
            type(input_adapter).__name__,
            type(output_adapter).__name__,
            device,
        )
        family = self._detect_model_family()
        print(f"Family: {family}")
        if family == "uma":
            return UMACalculator(
                model_patcher=self.model_patcher,
                input_adapter=input_adapter,
                output_adapter=output_adapter,
                properties=self.properties,
                dtype=self.dtype,
                device=device,
                keep_on_device=self.keep_on_device,
            )
        else:
            return MLPCalculator(
                model_patcher=self.model_patcher,
                input_adapter=input_adapter,
                output_adapter=output_adapter,
                properties=self.properties,
                dtype=self.dtype,
                device=device,
                keep_on_device=self.keep_on_device,
            )



    def _get_backbone(self) -> nn.Module:
        model = self.model_patcher.model
        return model.diffusion_model if hasattr(model, "diffusion_model") else model

    def _detect_model_family(self) -> str:

        backbone = self._get_backbone()
        cls_name = type(backbone).__name__.lower()
        if "escnmd" in cls_name or "uma" in cls_name:
            return "uma"

        if hasattr(backbone, "otf_graph") and hasattr(backbone, "cutoff"):
            return "uma"

        if "tensornet" in cls_name:
            return "tensornet"
        if "torchmd" in cls_name or "equivariant" in cls_name:
            return "torchmdnet"


        return "simple"

    def _resolve_input_adapter(self) -> InputAdapter:

        family = self._detect_model_family()

        if family == "uma":
            return UMAInputAdapter(task=self.task)

        # TensorNet, NewtonNet, etc. — all use simple (z, pos, batch)
        needs_grad = "forces" in self.properties
        return SimpleInputAdapter(needs_grad=needs_grad)

    def _resolve_output_adapter(self) -> OutputAdapter:
        family = self._detect_model_family()
        if family == "uma":
            return UMAOutputAdapter(self.properties)

        return SimpleOutputAdapter(self.properties)

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device) if isinstance(self.device, str) else self.device
        return self.model_patcher.load_device



    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> CalculatorBuilder:

        from mlpui.mlp import load_checkpoint_guess_config
        patcher = load_checkpoint_guess_config(ckpt_path)
        return cls(model_patcher=patcher, **kwargs)

    @classmethod
    def from_patcher_with_patches(
        cls,
        base_patcher,
        finetune_sd: dict[str, torch.Tensor],
        **kwargs,
    ) -> CalculatorBuilder:

        cloned = base_patcher.clone()
        cloned.load_finetune(finetune_sd)
        return cls(model_patcher=cloned, **kwargs)


class MixedPBCError(ValueError):

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, but the atoms object has PBC set to True for some"
        "dimensions but not others. Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class AllZeroUnitCellError(ValueError):

    def __init__(
        self,
        message="Atoms object claims to have PBC set, but the unit cell is identically 0. Please ensure that the atoms"
        "object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)