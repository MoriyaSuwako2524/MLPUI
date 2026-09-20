# Bundled model sources

MLPUI includes these model implementations, rather than importing separately
installed `newtonnet` or `torchmdnet` Python packages. Numerical dependencies
(PyTorch, PyG, NumPy and lightning-utilities) remain external dependencies.

| Directory | Source | Revision | License |
| --- | --- | --- | --- |
| `newtonnet/` | https://github.com/MoriyaSuwako2524/NewtonNet | `b3214ea870d9c15cd07e4b970f113bdeaed2a69d` | Regents license, see `newtonnet/LICENSE` |
| `torchmdnet/` | https://github.com/MoriyaSuwako2524/torchmd-net | `c162e4f1ffcdef0baae9e283d92d7cf5a366b74b` | MIT, see `torchmdnet/LICENSE` |

The NewtonNet license grants use, copying, modification and distribution for
educational, research and not-for-profit purposes. The project's MIT declaration
does not relicense these files; retain each upstream license when distributing.

Included: NewtonNet model/layer Python sources; TorchMD-Net model, prior and
utility Python sources plus CPU/CUDA neighbor extension sources. Upstream
datasets, training applications, notebooks, weights and compiled binaries are
not included. The older `mlpui/models/tensornet` directory is not the backend used
by the unified trainer; TensorNet is provided by `torchmdnet/models/tensornet.py`.

Integration changes:

- Python imports use the `mlpui.models` namespace without global module aliases.
- NewtonNet imports `les` only for BEC or for charge-before-energy configurations
  that use LES. A charge head after energy provides independent charge prediction.
- Native neighbor operators use the distinct `mlpui_torchmdnet_extensions`
  namespace to coexist with external packages. Native kernels remain optional.
- `torchmdnet/extensions/ops.py` provides a differentiable PyTorch fallback,
  following the upstream CPU pair ordering, periodic image and padding rules.
  It uses quadratic memory/time and does not support CUDA graph capture.
- Existing state dictionaries keep their parameter names. Explicitly trusted
  full-model pickle loading remaps historical Python module names to these
  implementations; missing historical classes still require source-side export.

Do not use `scripts/install_local_backends.py` for normal installation. That
legacy helper only prepares external reference packages for comparison tests.
