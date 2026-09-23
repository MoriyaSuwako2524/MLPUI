"""Reproduce the bundled MACE subset from the official 0.3.16 wheel.

Usage: python -m scripts.vendor_mace path/to/mace_torch-0.3.16-py3-none-any.whl
"""
import hashlib
from pathlib import Path
import re
import sys
import zipfile

WHEEL_SHA256 = "b80407edf6b2a1ec8523668c2a36852d20927ce1c3c56b70983a9f2dc53233ad"
FILES = {
    "modules": ("models", "blocks", "embeddings", "gate", "irreps_tools", "radial",
                "symmetric_contraction", "utils", "wrapper_ops"),
    "tools": ("cg", "compile", "scatter", "torch_tools", "utils"),
}


def main(wheel):
    wheel = Path(wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("Expected the official mace-torch 0.3.16 wheel (SHA256 mismatch)")
    target = Path(__file__).resolve().parents[1] / "mlpui/models/mace"
    with zipfile.ZipFile(wheel) as archive:
        for folder, names in FILES.items():
            (target / folder).mkdir(parents=True, exist_ok=True)
            for name in names:
                text = archive.read(f"mace/{folder}/{name}.py").decode()
                text = re.sub(r"\bfrom mace\b", "from mlpui.models.mace", text)
                # Batch is used only in utility annotations/statistics; reuse installed PyG.
                text = text.replace("from mlpui.models.mace.tools.torch_geometric.batch import Batch",
                                    "from torch_geometric.data import Batch")
                (target / folder / f"{name}.py").write_text(text, encoding="utf-8", newline="\n")
        (target / "LICENSE.md").write_bytes(archive.read("mace_torch-0.3.16.dist-info/licenses/LICENSE.md"))
    # __init__.py and mlpui_model.py are maintained MLPUI integration files.
    (target / "modules/__init__.py").write_text('"""Model components; upstream training/Polar modules are not bundled."""\n', encoding="utf-8")
    (target / "tools/__init__.py").write_text('"""Numerical utilities only; no upstream training application imports."""\nfrom .torch_tools import to_numpy\n', encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
