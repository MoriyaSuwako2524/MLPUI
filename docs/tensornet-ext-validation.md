# TensorNet-ext replacement acceptance (2026-10-18)

Implementation base: MLPUI `f7536676e02e1a64d1f0c9be9042e27b24d836db`.
Pinned upstream: panxl/torchmd-net `b0f7c7b42c6304c016a9e3dee78d4e2f4cbadc7d`.
The unmodified test fixture SHA256 is
`95c9507cd5ceedcb8700229a16286450ca2b2c8043f8b5deb7fd87ecaa3fe9ce`.

## Automated checks

Python 3.13.7, PyTorch 2.10.0+cpu; one CPU thread for scientific tests.

- 47 tests passed across `test_tensornet_ext.py`, `test_qmmm.py`,
  `test_qmmm_gesp.py`, and `test_qmmm_review.py`.
- 7 tests passed across `test_web_qmmm.py`, `test_qmmm_targets.py`, and
  `test_prepare_cm_fourtarget.py`. HTTP tests use isolated random-port localhost
  servers and synthetic six-frame data. Both baseline modes, optional GESP,
  custom external cutoff, upload/split/train/download/reload/evaluate/export pass.
- 7 shared backend regression tests passed (`test_backends.py`), after installing
  the repository-declared `lightning-utilities` dependency in the workspace venv.
- 12 Node tests passed (`node --test tests/web_ui.test.cjs`).

The new tests compare the full representation and E/F/ESP/GESP against the
pinned upstream source with identical weights (float64 tolerance 2e-12 absolute,
2e-11 relative). They test two interaction layers, ragged QM/MM batches against
individual frames, overlapping different frames, explicit MM batches and offsets,
batched offline composition, empty regions, isolated trailing QM sites, one-edge
shapes, external cutoff boundary, O(3)/translations/permutations, finite-difference
energy/charge/force derivatives, fixed-response kernel derivatives, zero-charge
continuity, and GESP-loss parameter finite differences in the external kernel,
QM interaction and scalar head. They also distinguish GESP from the total mixed
charge/position derivative and reject old state dictionaries and trusted full
modules. Existing checkpoint/CLI/four-target backward tests pass.

## Existing real-frame smoke

Read-only input: the previously prepared CM four-target **train** dataset.
No labels, splits, loss weights, source data, short-contact flags or old results
were changed. No optimizer steps, epochs, formal job submissions, service
restarts or pushes were performed.

| source_index | QM atoms | MM points | short_contact |
|---|---:|---:|---|
| 0 | 24 | 899 | false |
| 2059 | 24 | 906 | true |
| 11348 | 24 | 902 | true |

All three frames pass four-target loss/backward with finite parameter gradients.
The resource-limited backward model uses H=16, 2 layers, 8 QM/external RBFs,
float32 and residual-only labels. The default H=128/2-layer/32-RBF model also
passes a four-target forward on source_index 0 (about 0.57 s on this host).
These are initialization/derivative checks, not trained-accuracy measurements.

The exact existing loss weights used in the smoke are energy=0.18702975849502435,
forces=3.648907233023751, ESP=34.470263794659196,
GESP=363.5645895864644. The script reads them from the earlier training request.
`state_dict` is checked unchanged after backward. The old best100 checkpoint
SHA256 remains `3ab38b64bab9bc8a56e3e2e4d49be304e517bb5df86b701f47a76ed479830614`;
loading it raises `Incompatible QMMM architecture`.

Reproduce with `python -m scripts.smoke_tensornet_ext --dataset TRAIN_DIR
--training-request OLD_REQUEST_JSON --legacy-checkpoint OLD_BEST100
--output REPORT_JSON`. The script only reads those inputs and writes its report.
It rejects selected frame IDs outside the existing per-window training range.

CPU fallback neighbor search was exercised; CUDA compiled-neighbor performance
and long-run training accuracy were not evaluated. Finite 10 A data selection,
no PBC, offline-baseline derivative semantics and full dq/dR QM-force baseline
requirements remain as documented in `tensornet-qmmm.md`.
