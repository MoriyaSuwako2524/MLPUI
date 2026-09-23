"""ASE adapters for registered model backends, with explicit unit conversion."""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress

from mlpui.calculator import InputAdapter, OutputAdapter, MLPCalculator
from mlpui.backends import get_backend
from mlpui.backends.base import OUTPUT_ALIASES


class AtomicInputAdapter(InputAdapter):
    def __init__(self, family, length_to_angstrom=1.0, charge=None, spin=0, require_charge=False, model=None):
        self.family = family
        self.backend = get_backend(family)
        self.length_to_angstrom = length_to_angstrom
        self.charge = charge
        self.spin = spin
        self.require_charge = require_charge
        self.model = model

    def convert(self, atoms, device, dtype):
        return self.backend.prepare_inputs(
            atoms, device, dtype, length_to_angstrom=self.length_to_angstrom,
            charge=self.charge, spin=self.spin, require_charge=self.require_charge, model=self.model)


class AtomicOutputAdapter(OutputAdapter):
    ALIASES = OUTPUT_ALIASES

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
        self.backend = get_backend(family)
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
        available = self.backend.available_properties(model)
        missing = set(self._properties) - available
        if missing:
            raise PropertyNotImplementedError(f"{self.family} checkpoint does not provide {sorted(missing)}")

    def _forward(self, model_input):
        model = self._get_backbone()
        return self.backend.forward(model, model_input, self._properties)


def build_external_calculator(builder, family, device):
    factors = (builder.energy_to_ev, builder.length_to_angstrom, builder.dipole_to_eangstrom)
    if not all(np.isfinite(v) and v > 0 for v in factors):
        raise ValueError("Unit conversion factors must be finite and positive")
    dtype = builder.dtype or builder.model_patcher.get_dtype()
    model = builder._get_backbone()
    return ExternalCalculator(
        family=family, model_patcher=builder.model_patcher,
        input_adapter=AtomicInputAdapter(family, builder.length_to_angstrom, builder.charge, builder.spin,
                                         require_charge=getattr(model, "charge_constraint", False), model=model),
        output_adapter=AtomicOutputAdapter(builder.properties, *factors),
        properties=builder.properties, dtype=dtype, device=device,
        keep_on_device=builder.keep_on_device,
    )
