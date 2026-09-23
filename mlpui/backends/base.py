"""Backend contract; orchestration never needs to know a model's native API."""
from abc import ABC, abstractmethod
import copy

import numpy as np
import torch


OUTPUT_ALIASES = {
    "energy": ("energy", "y"),
    "free_energy": ("energy", "y"),
    "forces": ("forces", "neg_dy", "gradient_force", "direct_force"),
    "dipole": ("dipole", "vec"),
    "charges": ("charges", "charge"),
    "stress": ("stress",),
}


def normalize_output(output):
    if isinstance(output, tuple):
        output = dict(zip(("energy", "forces"), output))
    elif not isinstance(output, dict):
        output = vars(output)
    return {name: next((output[k] for k in keys if output.get(k) is not None), None)
            for name, keys in OUTPUT_ALIASES.items()}


class ModelBackend(ABC):
    """Stateless adapter. Model instances, weights and device state stay outside it.

    Metadata is a backend's potential capability, while available_properties()
    inspects the actual model. Methods must not import model packages until used.
    """

    name: str
    label: str
    aliases = ()
    config_container = None
    config_version = 1
    training_targets = ("energy", "forces", "charges")

    def metadata(self):
        return dict(label=self.label, config_container=self.config_container,
                    training_targets=list(self.training_targets))

    def settings(self, config):
        if self.config_container and isinstance(config.get(self.config_container), dict):
            return config[self.config_container]
        return config

    def configure_targets(self, config, targets):
        config = copy.deepcopy(config)
        if not isinstance(self.settings(config).get("charge_constraint", False), bool):
            raise ValueError("charge_constraint must be boolean")
        if "charges" in targets:
            self.add_charge_head(self.settings(config))
        return config

    def training_config(self, config, targets, dtype):
        return copy.deepcopy(self.settings(self.configure_targets(config, targets)))

    @abstractmethod
    def default_config(self): ...

    @abstractmethod
    def add_charge_head(self, config): ...

    @abstractmethod
    def matches(self, state): ...

    @abstractmethod
    def build_model(self, config): ...

    @abstractmethod
    def load_model(self, state, config): ...

    @abstractmethod
    def validate_training(self, model, config, targets): ...

    @abstractmethod
    def available_properties(self, model): ...

    def validate_constraint(self, model):
        if "charges" not in self.available_properties(model):
            raise ValueError("Total-charge constraint requires a charge output head")

    def geometry(self, atoms, device, dtype, length_to_angstrom, charge, require_charge):
        if require_charge and charge is None:
            raise ValueError("Total-charge constraint requires explicit Q (CalculatorBuilder charge=...)")
        if charge is not None and not np.isfinite(charge):
            raise ValueError("Total charge Q must be finite")
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise ValueError("These backends support nonperiodic or fully periodic cells; mixed PBC is unsupported")
        periodic = bool(np.all(atoms.pbc))
        if periodic and abs(np.linalg.det(atoms.cell.array)) < 1e-12:
            raise ValueError("Periodic calculations require a nonsingular cell")
        cell = atoms.cell.array / length_to_angstrom if periodic else np.zeros((3, 3))
        data = {
            "z": torch.as_tensor(atoms.numbers, dtype=torch.long, device=device),
            "pos": torch.tensor(atoms.positions / length_to_angstrom, dtype=dtype, device=device),
            "batch": torch.zeros(len(atoms), dtype=torch.long, device=device),
        }
        return data, cell, periodic

    @abstractmethod
    def prepare_inputs(self, atoms, device, dtype, *, length_to_angstrom=1.,
                       charge=None, spin=0, require_charge=False, model=None):
        """Convert inputs; model provides element tables/cutoffs for graph backends."""
        ...

    @abstractmethod
    def forward(self, model, inputs, properties):
        """Return canonical property names, retaining autograd when required."""
        ...
