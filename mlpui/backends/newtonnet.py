"""NewtonNet-specific configuration, serialization and execution."""
import numpy as np
import torch
from torch import nn

from .base import ModelBackend, OUTPUT_ALIASES, normalize_output


class NewtonNetBackend(ModelBackend):
    name = "newtonnet"
    label = "NewtonNet"
    config_container = "model"
    training_targets = ("energy", "forces", "charges", "dipole", "stress")

    def default_config(self):
        return dict(cutoff=5., n_features=64, n_basis=20, n_interactions=3,
                    activation="silu", output_properties=["energy", "gradient_force"])

    def add_charge_head(self, config):
        outputs = list(config.get("output_properties", []))
        if "charge" not in outputs:
            outputs.append("charge")
        config["output_properties"] = outputs

    def matches(self, state):
        if isinstance(state, nn.Module):
            return type(state).__module__.startswith(("newtonnet.", "mlpui.models.newtonnet."))
        return "embedding_layers.node_embedding.weight" in state

    def build_model(self, config):
        from mlpui.models.newtonnet.models.newtonnet import NewtonNet
        return NewtonNet(**self.settings(config))

    def load_model(self, state, config):
        config = self.settings(config)
        if not config or "output_properties" not in config:
            raise ValueError(
                "NewtonNet state_dict needs model_config with output_properties, "
                "cutoff and architecture settings (or the training config.yml). "
                "These cannot be inferred reliably from tensor shapes.")
        config = dict(config)
        config.pop("pretrained_model", None)
        model = self.build_model(config)
        dtype = next((v.dtype for v in state.values() if v.is_floating_point()), torch.float32)
        model.to(dtype=dtype)
        nn.Module.load_state_dict(model, state, strict=True)
        return model

    def validate_training(self, model, config, targets):
        if set(model.output_properties).intersection({"hessian", "bec"}):
            raise ValueError("Hessian/BEC heads require specialized training")

    def available_properties(self, model):
        outputs = set(model.output_properties)
        if outputs.intersection({"hessian", "bec"}):
            raise ValueError("NewtonNet checkpoints with hessian/bec heads require a specialized calculator")
        return {p for p, names in OUTPUT_ALIASES.items() if outputs.intersection(names)}

    def validate_constraint(self, model):
        if "charge" not in model.output_properties:
            raise ValueError("Total-charge constraint requires a charge output head")

    def prepare_inputs(self, atoms, device, dtype, *, length_to_angstrom=1.,
                       charge=None, spin=0, require_charge=False, model=None):
        data, cell, periodic = self.geometry(atoms, device, dtype, length_to_angstrom, charge, require_charge)
        if periodic and not np.allclose(atoms.cell.array, np.diag(atoms.cell.array.diagonal())):
            raise ValueError("This NewtonNet backend requires orthorhombic periodic cells; triclinic cells are unsupported")
        data["cell"] = torch.tensor(cell, dtype=dtype, device=device).unsqueeze(0)
        if charge is not None:
            data["q"] = torch.tensor([charge], dtype=dtype, device=device)
        return data

    def forward(self, model, inputs, properties):
        # Even an energy-only request may execute derivative heads.
        with torch.enable_grad():
            return normalize_output(model(**inputs))
