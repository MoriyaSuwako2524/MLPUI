"""MLPUI's supervised-charge extension of the pinned ScaleShiftMACE.

The upstream energy model is unchanged. Only invariant final-layer scalars enter
the independent charge head; its outputs do not alter energy or gradient forces.
"""
import threading

import numpy as np
import torch
from torch import nn
from e3nn import o3

from mlpui.models.charge import constrain_charges
from .modules.models import ScaleShiftMACE
from .modules.blocks import RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock


_construction_lock = threading.RLock()


class MACEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.charge_constraint = config["charge_constraint"]
        self.scalar_channels = config["num_channels"]
        self.source_version = "0.3.16"
        # e3nn/MACE initialize constants in the global default dtype. Restore it
        # even if construction fails; serialize our own concurrent constructors.
        dtype = torch.float64 if config["precision"] == 64 else torch.float32
        with _construction_lock:
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                hidden = o3.Irreps([(self.scalar_channels, (ell, (-1) ** ell))
                                   for ell in range(config["max_L"] + 1)])
                self.core = ScaleShiftMACE(
                    r_max=config["r_max"], num_bessel=config["num_bessel"],
                    num_polynomial_cutoff=config["num_polynomial_cutoff"],
                    max_ell=config["max_ell"],
                    interaction_cls_first=RealAgnosticInteractionBlock,
                    interaction_cls=RealAgnosticResidualInteractionBlock,
                    num_interactions=config["num_interactions"],
                    num_elements=len(config["atomic_numbers"]), hidden_irreps=hidden,
                    MLP_irreps=o3.Irreps(f'{config["readout_channels"]}x0e'),
                    atomic_energies=np.array([config["atomic_energies"][str(z)]
                                             for z in config["atomic_numbers"]]),
                    avg_num_neighbors=config["avg_num_neighbors"],
                    atomic_numbers=config["atomic_numbers"], correlation=config["correlation"],
                    gate=torch.nn.functional.silu, radial_MLP=config["radial_MLP"],
                    atomic_inter_scale=config["atomic_inter_scale"],
                    atomic_inter_shift=config["atomic_inter_shift"],
                    # Pure e3nn path, independent of optional accelerator installs.
                    use_reduced_cg=False,
                )
                self.charge_head = (nn.Sequential(
                    nn.Linear(self.scalar_channels, config["charge_hidden_channels"]), nn.SiLU(),
                    nn.Linear(config["charge_hidden_channels"], 1)) if config["predict_charges"] else None)
            finally:
                torch.set_default_dtype(previous)

    def forward(self, data, *, compute_force=True, compute_stress=False, q=None):
        output = self.core(data, training=self.training, compute_force=compute_force,
                           compute_stress=compute_stress)
        if self.charge_head is not None:
            # Upstream's final product block contains only num_channels x 0e.
            scalars = output["node_feats"][:, -self.scalar_channels:]
            charges = self.charge_head(scalars)
            if self.charge_constraint:
                charges = constrain_charges(charges, data["batch"], q)
            output["charges"] = charges
        return output
