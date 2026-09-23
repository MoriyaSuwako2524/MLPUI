"""Bundled MACE 0.3.16 with an optional supervised atomic-charge head."""
import copy
import math
from numbers import Real

import numpy as np
import torch
from torch import nn

from .base import ModelBackend, normalize_output


class MACEBackend(ModelBackend):
    name = "mace"
    label = "MACE"
    training_targets = ("energy", "forces", "charges", "stress")

    def metadata(self):
        return {**super().metadata(), "config_hint":
                "请将 atomic_numbers 设为数据包含的元素，并填写对应 atomic_energies（训练单位）。默认参考能为 0，不会自动拟合；继续训练须使用原模型配置。"}

    def configure_targets(self, config, targets):
        return self.validate_config(super().configure_targets(config, targets))

    def default_config(self):
        elements = [1, 6, 7, 8, 9, 15, 16, 17]
        return dict(source_version="0.3.16", precision=32, atomic_numbers=elements,
            atomic_energies={str(z): 0. for z in elements}, r_max=5., num_bessel=8,
            num_polynomial_cutoff=5, max_ell=3, num_channels=64, max_L=1,
            num_interactions=2, correlation=3, readout_channels=16,
            radial_MLP=[64, 64, 64], avg_num_neighbors=8.,
            atomic_inter_scale=1., atomic_inter_shift=0.,
            predict_charges=False, charge_hidden_channels=32, charge_constraint=False)

    def add_charge_head(self, config):
        config["predict_charges"] = True

    def validate_config(self, config):
        defaults = self.default_config()
        if set(config) - defaults.keys():
            raise ValueError(f"Unsupported MACE settings: {sorted(set(config) - defaults.keys())}")
        config = {**defaults, **copy.deepcopy(config)}
        if config["source_version"] != "0.3.16":
            raise ValueError("This backend bundles MACE source_version 0.3.16")
        if type(config["precision"]) is not int or config["precision"] not in (32, 64):
            raise ValueError("MACE precision must be 32 or 64")
        elements = config["atomic_numbers"]
        if (not isinstance(elements, list) or not elements
                or any(type(z) is not int or not 1 <= z <= 118 for z in elements)
                or elements != sorted(set(elements))):
            raise ValueError("MACE atomic_numbers must be sorted, unique atomic numbers (1-118)")
        energies = config["atomic_energies"]
        if not isinstance(energies, dict):
            raise ValueError("MACE atomic_energies must map every atomic number to its reference energy")
        energies = {str(key): value for key, value in energies.items()}
        if set(energies) != {str(z) for z in elements}:
            raise ValueError("MACE atomic_energies must match atomic_numbers exactly")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in energies.values()):
            raise ValueError("MACE atomic_energies must be finite numbers")
        config["atomic_energies"] = energies
        for key in ("num_bessel", "num_polynomial_cutoff", "num_channels", "num_interactions",
                    "correlation", "readout_channels", "charge_hidden_channels"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"MACE {key} must be a positive integer")
        for key in ("max_L", "max_ell"):
            if type(config[key]) is not int or not 0 <= config[key] <= 3:
                raise ValueError(f"MACE {key} must be an integer between 0 and 3")
        if config["max_L"] > config["max_ell"] or config["correlation"] > 3:
            raise ValueError("MACE requires max_L <= max_ell and correlation <= 3")
        for key in ("r_max", "avg_num_neighbors", "atomic_inter_scale", "atomic_inter_shift"):
            value = config[key]
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    or (key != "atomic_inter_shift" and value <= 0)):
                raise ValueError(f"MACE {key} must be finite" + (" and positive" if key != "atomic_inter_shift" else ""))
        if (not isinstance(config["radial_MLP"], list) or not config["radial_MLP"]
                or any(type(n) is not int or n < 1 for n in config["radial_MLP"])):
            raise ValueError("MACE radial_MLP must contain positive integer widths")
        for key in ("predict_charges", "charge_constraint"):
            if not isinstance(config[key], bool):
                raise ValueError(f"MACE {key} must be boolean")
        if config["charge_constraint"] and not config["predict_charges"]:
            raise ValueError("Total-charge constraint requires a charge output head")
        return config

    def training_config(self, config, targets, dtype):
        config = super().training_config(config, targets, dtype)
        config["precision"] = 64 if dtype == torch.float64 else 32
        return self.validate_config(config)

    def matches(self, state):
        if isinstance(state, nn.Module):
            return type(state).__module__.startswith(("mace.", "mlpui.models.mace."))
        return "core.atomic_numbers" in state and any(k.startswith("core.products.") for k in state)

    def build_model(self, config):
        config = self.validate_config(config)
        try:
            from mlpui.models.mace.mlpui_model import MACEModel
        except ImportError as exc:
            raise ImportError("Bundled MACE requires e3nn==0.4.4 and opt_einsum_fx==0.1.4; reinstall MLPUI dependencies") from exc
        return MACEModel(config)

    def load_model(self, state, config):
        if not config:
            raise ValueError("MACE state_dict requires the matching model_config")
        config = self.validate_config(config)
        dtype = next((v.dtype for v in state.values() if v.is_floating_point()), torch.float32)
        config["precision"] = 64 if dtype == torch.float64 else 32
        model = self.build_model(config)
        nn.Module.load_state_dict(model, state, strict=True)
        # These buffers encode architecture/element semantics and must agree with
        # the configuration, not merely have the same shapes.
        if model.core.atomic_numbers.tolist() != config["atomic_numbers"]:
            raise ValueError("MACE checkpoint atomic_numbers disagree with model_config")
        return model

    def validate_training(self, model, config, targets):
        missing = set(targets) - self.available_properties(model)
        if missing:
            raise ValueError(f"MACE does not produce training targets {sorted(missing)}")

    def available_properties(self, model):
        from mlpui.models.mace.mlpui_model import MACEModel
        if not isinstance(model, MACEModel):
            raise ValueError("Use an MLPUI MACE checkpoint; importing upstream foundation/Polar models requires explicit conversion")
        available = {"energy", "free_energy", "forces", "stress"}
        if model.charge_head is not None:
            available.add("charges")
        return available

    def prepare_inputs(self, atoms, device, dtype, *, length_to_angstrom=1.,
                       charge=None, spin=0, require_charge=False, model=None):
        from ase.neighborlist import primitive_neighbor_list
        self.available_properties(model)
        data, cell, _ = self.geometry(atoms, device, dtype, length_to_angstrom, charge, require_charge)
        elements = model.core.atomic_numbers.to(device=device)
        membership = data["z"][:, None] == elements[None, :]
        if not membership.any(dim=1).all():
            unknown = sorted(set(atoms.numbers) - set(elements.cpu().tolist()))
            raise ValueError(f"MACE dataset contains atomic numbers {unknown} outside model atomic_numbers")
        positions = atoms.positions / length_to_angstrom
        i, j, shifts = primitive_neighbor_list(
            "ijS", atoms.pbc, cell, positions, float(model.core.r_max), self_interaction=False)
        graph = dict(
            positions=data["pos"], node_attrs=membership.to(dtype=dtype), batch=data["batch"],
            ptr=torch.tensor([0, len(atoms)], dtype=torch.long, device=device),
            edge_index=torch.tensor(np.stack((i, j)), dtype=torch.long, device=device),
            shifts=torch.tensor(shifts @ cell, dtype=dtype, device=device),
            unit_shifts=torch.tensor(shifts, dtype=dtype, device=device),
            cell=torch.tensor(cell, dtype=dtype, device=device).unsqueeze(0),
            head=torch.zeros(1, dtype=torch.long, device=device),
        )
        q = None if charge is None else torch.tensor([charge], dtype=dtype, device=device)
        return dict(data=graph, q=q)

    def forward(self, model, inputs, properties):
        # Evaluation of derivative properties also needs coordinate autograd.
        with torch.enable_grad():
            return normalize_output(model(**inputs, compute_force="forces" in properties,
                                          compute_stress="stress" in properties))
