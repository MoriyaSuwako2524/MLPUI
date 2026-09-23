"""TorchMD-Net (including TensorNet) configuration and execution."""
import copy
import re

import torch
from torch import nn

from .base import ModelBackend, normalize_output


class TorchMDNetBackend(ModelBackend):
    name = "torchmdnet"
    label = "TorchMD-Net / TensorNet"
    aliases = ("torchmd-net", "tensornet")
    training_targets = ("energy", "forces", "charges")

    def default_config(self):
        return dict(model="tensornet", precision=32, embedding_dimension=64,
            num_layers=3, num_rbf=32, rbf_type="expnorm", trainable_rbf=False,
            activation="silu", cutoff_lower=0., cutoff_upper=5., max_z=119,
            max_num_neighbors=64, aggr="add", neighbor_embedding=True,
            attn_activation="silu", num_heads=4, distance_influence="both",
            equivariance_invariance_group="O(3)", derivative=True, atom_filter=-1,
            prior_model=None, output_model="Scalar", reduce_op="sum",
            pred_dict={"y": 1., "neg_dy": 1.})

    def add_charge_head(self, config):
        heads = dict(config.get("pred_dict", {"y": 1., "neg_dy": 1.}))
        heads.setdefault("charge", 1.)
        config["pred_dict"] = heads

    def training_config(self, config, targets, dtype):
        config = super().training_config(config, targets, dtype)
        config["precision"] = 64 if dtype == torch.float64 else 32
        if "forces" in targets:
            config["derivative"] = True
        return config

    def matches(self, state):
        if isinstance(state, nn.Module):
            return type(state).__module__.startswith(("torchmdnet.", "mlpui.models.torchmdnet."))
        return any(k.startswith("representation_model.") for k in state)

    def build_model(self, config):
        from mlpui.models.torchmdnet.models.model import create_model
        return create_model(copy.deepcopy(config))

    def load_model(self, state, config):
        from mlpui.models.torchmdnet.models.model import TorchMD_Net
        if not config or "model" not in config:
            raise ValueError("TorchMD-Net needs checkpoint hyper_parameters or model_config")
        model = self.build_model(config)
        if any(k.startswith("output_model.") for k in state) and hasattr(model, "output_modules"):
            model = TorchMD_Net(
                model.representation_model, model.output_modules["y"],
                prior_model=model.prior_model, derivative=config.get("derivative", False),
                dtype=next(model.parameters()).dtype)
        patterns = [
            (r"output_model\.output_network\.(\d+)\.update_net\.(\d+)\.",
             r"output_model.output_network.\1.update_net.layers.\2."),
            (r"output_model\.output_network\.([02])\.(weight|bias)",
             r"output_model.output_network.layers.\1.\2"),
        ]
        for old, new in patterns:
            state = {re.sub(old, new, k): v for k, v in state.items()}
        if hasattr(model, "output_modules"):
            for prefix in ("mean", "std"):
                for head in model.output_modules:
                    key = f"{prefix}_{head}"
                    if key in state:
                        model.register_buffer(key, torch.empty_like(state[key]))
                if prefix not in state and all(f"{prefix}_{h}" in state for h in model.output_modules):
                    if prefix in model._buffers:
                        delattr(model, prefix)
        # Bypass the fork's permissive override, retaining strict compatibility.
        nn.Module.load_state_dict(model, state, strict=True)
        model.mlpui_output_model = config.get("output_model", "Scalar")
        return model

    def validate_training(self, model, config, targets):
        if config.get("output_model", "Scalar") != "Scalar":
            raise ValueError("Training requires a Scalar primary energy head")
        if "forces" in targets:
            model.derivative = True

    def available_properties(self, model):
        output_type = getattr(model, "mlpui_output_model", "Scalar")
        if output_type != "Scalar":
            raise ValueError(f"Expected an energy Scalar head, got {output_type!r}")
        available = {"energy", "free_energy", "forces"}
        heads = getattr(model, "output_modules", {})
        if "vec" in heads:
            available.add("dipole")
        if "charge" in heads:
            available.add("charges")
        return available

    def validate_constraint(self, model):
        if "charge" not in getattr(model, "output_modules", {}):
            raise ValueError("Total-charge constraint requires a charge output head")

    def prepare_inputs(self, atoms, device, dtype, *, length_to_angstrom=1.,
                       charge=None, spin=0, require_charge=False, model=None):
        data, cell, periodic = self.geometry(atoms, device, dtype, length_to_angstrom, charge, require_charge)
        data["box"] = torch.tensor(cell, dtype=dtype, device=device) if periodic else None
        data["q"] = torch.tensor([charge if charge is not None else 0], dtype=dtype, device=device)
        data["s"] = torch.tensor([spin], dtype=dtype, device=device)
        return data

    def forward(self, model, inputs, properties):
        for module in model.modules():
            if type(module).__name__ == "OptimizedDistance":
                module.use_periodic = inputs["box"] is not None
        if "forces" in properties:
            model.derivative = True
        needs_grad = model.training or bool(getattr(model, "derivative", False))
        with torch.enable_grad() if needs_grad else torch.no_grad():
            return normalize_output(model(**inputs))
