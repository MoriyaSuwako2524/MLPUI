"""Overlay local fork Python sources in an isolated test venv.

On Windows this retains the compiled neighbor extension from torchmd-net-cpu.
For production/GPU environments, build and install the fork normally instead.
"""
import argparse
from pathlib import Path
import shutil
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchmd-root", type=Path, required=True)
    parser.add_argument("--newtonnet-root", type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Activate a dedicated test virtual environment first")
    site = Path(sysconfig.get_paths()["purelib"])
    extension_dir = site / "torchmdnet" / "extensions"
    if not list(extension_dir.glob("*.pyd")) and not list(extension_dir.glob("*.so")):
        parser.error("Install a compatible torchmd-net-cpu wheel first")
    sources = [(args.torchmd_root / "torchmdnet", "torchmdnet"),
               (args.newtonnet_root / "newtonnet", "newtonnet")]
    for source, name in sources:
        if not (source / "__init__.py").is_file():
            parser.error(f"Package source not found: {source}")
    for source, name in sources:
        for path in source.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            destination = site / name / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        print(f"Installed test source overlay: {source.resolve()} -> {site / name}")


if __name__ == "__main__":
    main()
