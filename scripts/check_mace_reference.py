"""Compare bundled core against unmodified numerical source in the pinned wheel.

Run: python -m scripts.check_mace_reference path/to/mace_torch-0.3.16-py3-none-any.whl
The temporary reference package skips upstream application __init__ imports.
"""
import copy
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import types
import zipfile

import numpy as np
import torch
from ase import Atoms

from scripts.vendor_mace import WHEEL_SHA256
from mlpui.backends import get_backend


def main(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("Reference wheel checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="mlpui-mace-reference-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(path) as archive:
            # Extract just Python modules, with explicit containment validation.
            for entry in archive.namelist():
                if entry.startswith("mace/") and entry.endswith(".py"):
                    target = (root / entry).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError("Invalid reference wheel path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(entry))
        for name in ("mace", "mace.modules", "mace.tools"):
            package = types.ModuleType(name)
            package.__path__ = [str(root / name.replace(".", "/"))]
            sys.modules[name] = package
        backend = get_backend("mace")
        config = backend.default_config()
        config.update(num_channels=4, max_L=1, max_ell=1, correlation=2,
                      radial_MLP=[8], precision=64, predict_charges=True)
        bundled = backend.build_model(config)
        # e3nn is now initialized with MLPUI's scoped constants compatibility fix.
        tools = importlib.import_module("mace.tools")
        tools.to_numpy = importlib.import_module("mace.tools.torch_tools").to_numpy
        modules = importlib.import_module("mace.modules.models")
        blocks = importlib.import_module("mace.modules.blocks")
        from e3nn import o3
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            reference = modules.ScaleShiftMACE(
                r_max=config["r_max"], num_bessel=config["num_bessel"],
                num_polynomial_cutoff=config["num_polynomial_cutoff"], max_ell=1,
                interaction_cls_first=blocks.RealAgnosticInteractionBlock,
                interaction_cls=blocks.RealAgnosticResidualInteractionBlock,
                num_interactions=2, num_elements=len(config["atomic_numbers"]),
                hidden_irreps=o3.Irreps("4x0e + 4x1o"), MLP_irreps=o3.Irreps("16x0e"),
                atomic_energies=np.zeros(len(config["atomic_numbers"])),
                atomic_numbers=config["atomic_numbers"], avg_num_neighbors=8., correlation=2,
                gate=torch.nn.functional.silu, radial_MLP=[8], atomic_inter_scale=1.,
                atomic_inter_shift=0., use_reduced_cg=False)
        finally:
            torch.set_default_dtype(previous)
        reference.load_state_dict(bundled.core.state_dict(), strict=True)
        for periodic in (False, True):
            atoms = Atoms("OH2", positions=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                          cell=[[7., 0., 0.], [.5, 7., 0.], [.2, .3, 7.]], pbc=periodic)
            graph = backend.prepare_inputs(atoms, "cpu", torch.float64, model=bundled)["data"]
            bundled.core.zero_grad()
            reference.zero_grad()
            actual = bundled.core(copy.deepcopy(graph), training=True, compute_stress=periodic)
            expected = reference(copy.deepcopy(graph), training=True, compute_stress=periodic)
            keys = ["energy", "forces", "node_feats"] + (["stress"] if periodic else [])
            for key in keys:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for output in (actual, expected):
                (output["energy"].square().mean() + output["forces"].square().mean()).backward()
            for p, q in zip(bundled.core.parameters(), reference.parameters()):
                if p.grad is None or q.grad is None:
                    assert p.grad is q.grad is None
                else:
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        print("Official MACE 0.3.16 reference: exact E/F/features/stress and parameter-gradient agreement (gas and periodic).")


if __name__ == "__main__":
    main(sys.argv[1])
