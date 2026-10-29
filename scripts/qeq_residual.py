"""Subtract a trained QEq E_charge model from a dual-region dataset.

    python -m scripts.qeq_residual CHECKPOINT SOURCE_DIR OUTPUT_DIR [--batch-size 16] [--device cpu]

SOURCE_DIR is an NpyDataset whose energy/forces are environment labels
(E_QMMM - E_gas, F_QMMM - F_gas) and whose mm_potential / mm_potential_gradient
(/ mm_forces) are the QM/MM labels. OUTPUT_DIR receives the same files with
those targets replaced by label - QEq prediction, i.e. residual_only labels for
TensorNet-QMMM, plus qeq_* prediction arrays and residual_manifest.json
(checkpoint SHA256, source hashes, RMS before/after). The source is not modified.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig

SUBTRACT = ('energy', 'forces', 'mm_forces', 'mm_potential', 'mm_potential_gradient')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float64', choices=('float32', 'float64'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory must not exist')
    dtype = getattr(torch, args.dtype)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = state['model_config']
    trainer = Trainer('qeq_charge', config, TrainingConfig(dtype=dtype, device=args.device,
                      loss_weights={'energy': 1.0}), checkpoint=args.checkpoint)
    model, backend = trainer.model, trainer.backend
    model.eval()
    data = NpyDataset(args.source)
    targets = [k for k in SUBTRACT if k in data.arrays]
    predictions = {k: [] for k in targets + ['charges', 'gas_charges']}
    for start in range(0, len(data), args.batch_size):
        samples = [data[i] for i in range(start, min(start + args.batch_size, len(data)))]
        prepared = backend.prepare_batch(samples, args.device, dtype, model=model)
        inputs, layout = prepared
        out = backend.forward(model, inputs, targets)
        for key in predictions:
            predictions[key].append(out[key].detach().cpu().double().numpy())
    pred = {k: np.concatenate(v) for k, v in predictions.items()}
    args.output.mkdir(parents=True)
    summary = {}
    for path in sorted(args.source.glob('*.npy')):
        name = path.stem
        if name in targets:
            label = np.load(path).astype(np.float64)
            p = pred[name].reshape(label.shape)
            residual = label - p
            np.save(args.output / path.name, residual)
            np.save(args.output / f'qeq_{name}.npy', p)
            summary[name] = dict(label_rms=float(np.sqrt(np.mean(label ** 2))), residual_rms=float(np.sqrt(np.mean(residual ** 2))),
                                 label_std=float(label.std()), residual_std=float(residual.std()))
        else:
            shutil.copy2(path, args.output / path.name)
    np.save(args.output / 'qeq_charges.npy', pred['charges'])
    np.save(args.output / 'qeq_gas_charges.npy', pred['gas_charges'])
    manifest = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint), model_config=config,
                    source=str(args.source), source_sha256={p.name: sha256(p) for p in sorted(args.source.glob('*.npy'))},
                    subtracted=targets, summary=summary,
                    definition='residual = label - E_charge prediction (QEq, Hellmann-Feynman derivatives; no dq/dR terms)')
    (args.output / 'residual_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
