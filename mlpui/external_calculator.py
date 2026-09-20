"""ASE adapters for TorchMD-Net and NewtonNet, with explicit unit conversion."""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress

from mlpui.calculator import InputAdapter, OutputAdapter, MLPCalculator


class AtomicInputAdapter(InputAdapter):
    def __init__(self, family, length_to_angstrom=1.0, charge=None, spin=0, require_charge=False):
        self.family = family
        self.length_to_angstrom = length_to_angstrom
        self.charge = charge
        self.spin = spin
        self.require_charge = require_charge

    def convert(self, atoms, device, dtype):
        if self.require_charge and self.charge is None:
            raise ValueError("Total-charge constraint requires explicit Q (CalculatorBuilder charge=...)")
        if self.charge is not None and not np.isfinite(self.charge):
            raise ValueError("Total charge Q must be finite")
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise ValueError("These backends support nonperiodic or fully periodic cells; mixed PBC is unsupported")
        periodic = bool(np.all(atoms.pbc))
        if periodic and abs(np.linalg.det(atoms.cell.array)) < 1e-12:
            raise ValueError("Periodic calculations require a nonsingular cell")
        if periodic and self.family == "newtonnet" and not np.allclose(atoms.cell.array, np.diag(atoms.cell.array.diagonal())):
            raise ValueError("This NewtonNet backend requires orthorhombic periodic cells; triclinic cells are unsupported")
        cell = atoms.cell.array / self.length_to_angstrom if periodic else np.zeros((3, 3))
        data = {
            "z": torch.as_tensor(atoms.numbers, dtype=torch.long, device=device),
            "pos": torch.tensor(atoms.positions / self.length_to_angstrom, dtype=dtype, device=device),
            "batch": torch.zeros(len(atoms), dtype=torch.long, device=device),
        }
        if self.family == "newtonnet":
            data["cell"] = torch.tensor(cell, dtype=dtype, device=device).unsqueeze(0)
            if self.charge is not None:
                data["q"] = torch.tensor([self.charge], dtype=dtype, device=device)
        else:
            data["box"] = torch.tensor(cell, dtype=dtype, device=device) if periodic else None
            data["q"] = torch.tensor([self.charge if self.charge is not None else 0], dtype=dtype, device=device)
            data["s"] = torch.tensor([self.spin], dtype=dtype, device=device)
        return data


class AtomicOutputAdapter(OutputAdapter):
    ALIASES = {
        "energy": ("energy", "y"),
        "free_energy": ("energy", "y"),
        "forces": ("forces", "neg_dy", "gradient_force", "direct_force"),
        "dipole": ("dipole", "vec"),
        "charges": ("charges", "charge"),
        "stress": ("stress",),
    }

    def __init__(self, properties, energy_to_ev=1.0, length_to_angstrom=1.0, dipole_to_eangstrom=1.0):
        self.properties = properties
        self.factors = {
            "energy": energy_to_ev, "free_energy": energy_to_ev,
            "forces": energy_to_ev / length_to_angstrom,
            "stress": energy_to_ev / length_to_angstrom ** 3,
            "dipole": dipole_to_eangstrom, "charges": 1.0,
        }

    def convert(self, output, atoms):
        if isinstance(output, tuple):
            output = dict(zip(("energy", "forces"), output))
        if not isinstance(output, dict):
            output = vars(output)
        shapes = {"energy": (), "free_energy": (), "forces": (len(atoms), 3),
                  "dipole": (3,), "charges": (len(atoms),)}
        results = {}
        for prop in self.properties:
            value = next((output[k] for k in self.ALIASES[prop]
                          if k in output and output[k] is not None), None)
            if value is None or (isinstance(value, torch.Tensor) and value.numel() == 0):
                raise PropertyNotImplementedError(f"Model did not produce requested property {prop!r}")
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value) * self.factors[prop]
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {prop} returned by model")
            if prop == "stress":
                if abs(np.linalg.det(atoms.cell.array)) < 1e-12:
                    raise ValueError("Stress requires a nonsingular cell")
                value = full_3x3_to_voigt_6_stress(value.reshape(3, 3)) if value.size == 9 else value.reshape(6)
            else:
                if prop == "dipole" and value.size != 3:
                    raise PropertyNotImplementedError(
                        "This dipole head returns a magnitude, not a 3-vector; "
                        "it cannot be exposed as an ASE dipole vector."
                    )
                value = value.reshape(shapes[prop])
                if prop in ("energy", "free_energy"):
                    value = float(value)
            results[prop] = value
        return results


class ExternalCalculator(MLPCalculator):
    def __init__(self, *, family, **kwargs):
        self.family = family
        super().__init__(**kwargs)
        self.implemented_properties = list(self._properties)
        self._get_backbone().to(dtype=self._dtype)
        self._validate_properties()

    def _ensure_ready(self):
        # Cloned patchers share a model; another calculator may have offloaded it.
        self._ready = False
        super()._ensure_ready()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        from ase.calculators.calculator import all_changes
        current_atoms = atoms if atoms is not None else self.atoms
        if "stress" in self._properties and not np.all(current_atoms.pbc):
            raise ValueError("Stress requires a fully periodic cell")
        super().calculate(atoms, properties, all_changes if system_changes is None else system_changes)

    def _validate_properties(self):
        model = self._get_backbone()
        if self.family == "torchmdnet":
            output_type = getattr(model, "mlpui_output_model", "Scalar")
            if output_type != "Scalar":
                raise ValueError(f"Expected an energy Scalar head, got {output_type!r}")
            available = {"energy", "free_energy", "forces"}
            heads = getattr(model, "output_modules", {})
            if "vec" in heads:
                available.add("dipole")
            if "charge" in heads:
                available.add("charges")
        else:
            outputs = set(model.output_properties)
            available = {p for p, names in AtomicOutputAdapter.ALIASES.items() if outputs.intersection(names)}
            # Higher derivatives need additional autograd setup beyond this adapter.
            if outputs.intersection({"hessian", "bec"}):
                raise ValueError("NewtonNet checkpoints with hessian/bec heads require a specialized calculator")
        missing = set(self._properties) - available
        if missing:
            raise PropertyNotImplementedError(f"{self.family} checkpoint does not provide {sorted(missing)}")

    def _forward(self, model_input):
        model = self._get_backbone()
        if self.family == "torchmdnet":
            # ASE's current PBC flag takes precedence over a training-time box.
            for module in model.modules():
                if type(module).__name__ == "OptimizedDistance":
                    module.use_periodic = model_input["box"] is not None
        if self.family == "torchmdnet" and "forces" in self._properties:
            model.derivative = True
        # A model can compute derivative heads even for an energy-only request.
        needs_grad = bool(getattr(model, "derivative", False)) or self.family == "newtonnet"
        with torch.enable_grad() if needs_grad else torch.no_grad():
            return model(**model_input)


def build_external_calculator(builder, family, device):
    factors = (builder.energy_to_ev, builder.length_to_angstrom, builder.dipole_to_eangstrom)
    if not all(np.isfinite(v) and v > 0 for v in factors):
        raise ValueError("Unit conversion factors must be finite and positive")
    dtype = builder.dtype or builder.model_patcher.get_dtype()
    return ExternalCalculator(
        family=family, model_patcher=builder.model_patcher,
        input_adapter=AtomicInputAdapter(family, builder.length_to_angstrom, builder.charge, builder.spin,
                                         require_charge=getattr(builder.model_patcher.model, "charge_constraint", False)),
        output_adapter=AtomicOutputAdapter(builder.properties, *factors),
        properties=builder.properties, dtype=dtype, device=device,
        keep_on_device=builder.keep_on_device,
    )
