"""Build bundled neighbor kernels: python scripts/build_model_extensions.py [--cuda]."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Build CPU/CUDA kernels; requires matching CUDA toolkit")
    args = parser.parse_args()
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    os.chdir(Path(__file__).resolve().parents[1])
    source = Path("mlpui/models/torchmdnet/extensions")
    sources = [source / "torchmdnet_extensions.cpp", source / "neighbors/neighbors_cpu.cpp"]
    if args.cuda:
        sources.append(source / "neighbors/neighbors_cuda.cu")
    extension = (CUDAExtension if args.cuda else CppExtension)(
        name="mlpui.models.torchmdnet.extensions.mlpui_torchmdnet_extensions",
        sources=[str(path) for path in sources],
        define_macros=[("WITH_CUDA", 1)] if args.cuda else [],
    )
    setup(ext_modules=[extension], cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
          script_args=["build_ext", "--inplace"])


if __name__ == "__main__":
    main()
