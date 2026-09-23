# Bundled model sources

MLPUI includes these model implementations, rather than importing separately
installed `newtonnet` or `torchmdnet` Python packages. Numerical dependencies
(PyTorch, PyG, NumPy and lightning-utilities) remain external dependencies.

| Directory | Source | Revision | License |
| --- | --- | --- | --- |
| `newtonnet/` | https://github.com/MoriyaSuwako2524/NewtonNet | `b3214ea870d9c15cd07e4b970f113bdeaed2a69d` | Regents license, see `newtonnet/LICENSE` |
| `torchmdnet/` | https://github.com/MoriyaSuwako2524/torchmd-net | `c162e4f1ffcdef0baae9e283d92d7cf5a366b74b` | MIT, see `torchmdnet/LICENSE` |
| `mace/` | https://github.com/ACEsuit/mace/tree/v0.3.16 | Official `mace-torch==0.3.16` wheel | MIT, see `mace/LICENSE.md` |

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
- Optional `charge_constraint` projects predicted atomic charges to an explicitly
  supplied total Q per structure. It runs after scaling, before downstream
  NewtonNet outputs, and is stored with model configuration (no new parameters).

Do not use `scripts/install_local_backends.py` for normal installation. That
legacy helper only prepares external reference packages for comparison tests.

## MACE source and local integration

The exact source artifact is `mace_torch-0.3.16-py3-none-any.whl` from PyPI,
SHA256 `b80407edf6b2a1ec8523668c2a36852d20927ce1c3c56b70983a9f2dc53233ad`.
`python -m scripts.vendor_mace <wheel>` verifies that hash and reproduces the
14 upstream numerical Python modules and license. The integration files
`mace/__init__.py` and `mace/mlpui_model.py` are maintained separately.

- Includes the standard MACE/ScaleShiftMACE model components and required
  numerical utilities. Upstream training applications, datasets, calculators,
  foundation weights and PolarMACE are not included.
- Imports are rewritten into `mlpui.models.mace`; no global module aliases are
  installed. Package initializers omit upstream training/application imports.
  The `Batch` type used in numerical utility annotations is supplied by the
  existing PyG dependency instead of upstream's private PyG copy.
- `MACEModel` composes the unchanged ScaleShiftMACE core with an optional MLP
  over the final `0e` features for atomwise supervised charge prediction. The
  existing total-charge projection is applied only when requested. Neither
  charge prediction nor this projection adds an energy term.
- Graphs use ASE's neighbor list (including periodic image shifts); no matscipy
  or compiler is needed. The backend uses pure e3nn operations with reduced CG
  disabled, so weights do not depend on optional cuEquivariance installation.
- e3nn 0.4.4 and opt_einsum_fx 0.1.4 are pinned numerical dependencies. During
  e3nn's first import only, a scoped `safe_globals([slice])` allows its installed
  constants file to load under newer PyTorch. User checkpoint loading remains
  weights-only by default. Global default dtype is restored after construction.
- `scripts/check_mace_reference.py <wheel>` compares the original wheel's core
  against the bundled core for exact energy, forces, features, periodic stress
  and training parameter gradients using identical weights/graphs. It runs in a
  separate process and does not change production module resolution.
