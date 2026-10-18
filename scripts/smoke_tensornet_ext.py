"""Read-only real-frame forward/backward smoke; no optimizer steps or training jobs.

python -m scripts.smoke_tensornet_ext --dataset PATH --training-request JSON
    --legacy-checkpoint PATH --output JSON [--source-indices 0 2059 11348]
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from mlpui.backends import get_backend
from mlpui.data import NpyDataset
from mlpui.external_models import load_external_checkpoint
from mlpui.training import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--training-request', type=Path, required=True)
    parser.add_argument('--legacy-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-indices', nargs='+', type=int, default=[0,2059,11348])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    request = json.loads(args.training_request.read_text())
    weights = request['training']['loss_weights']
    props = tuple(weights)
    data = NpyDataset(args.dataset)
    ids = np.load(args.dataset/'source_index.npy', allow_pickle=False)
    if any(sid % 400 >= 280 for sid in args.source_indices):
        raise ValueError('Smoke restricted to existing train split (frame 0..279 per window)')
    backend = get_backend('tensornet_qmmm')
    c = backend.default_config(); c.update(embedding_dimension=16, num_layers=2, num_rbf=8, ext_num_rbf=8)
    trainer = Trainer('tensornet_qmmm', c, TrainingConfig(dtype=torch.float32, device='cpu', loss_weights=weights))
    before = {k:v.clone() for k,v in trainer.model.state_dict().items()}
    rows = []
    for sid in args.source_indices:
        matches = np.flatnonzero(ids == sid)
        if len(matches) != 1: raise ValueError(f'Expected exactly one source_index {sid}')
        s = data[int(matches[0])]
        start = time.monotonic()
        trainer.model.zero_grad(set_to_none=True)
        loss, parts = trainer._loss(s)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters())
        rows.append(dict(source_index=sid, qm_atoms=len(s['z']), mm_points=len(s['mm_pos']),
            short_contact=bool(s['short_contact']), loss=float(loss.detach()),
            target_mse={k:float(v.detach()) for k,v in parts.items()},
            parameter_grad_l1=sum(float(p.grad.abs().sum()) for p in trainer.model.parameters() if p.grad is not None),
            seconds=time.monotonic()-start))
        print(json.dumps(rows[-1]), flush=True)
    assert all(torch.equal(before[k],v) for k,v in trainer.model.state_dict().items())
    # Default-size model forward, separately from the limited-resource backward.
    model = backend.build_model(backend.default_config()).eval()
    s = data[int(np.flatnonzero(ids == args.source_indices[0])[0])]
    inputs = backend.prepare_sample(s, 'cpu', torch.float32, model=model)
    start = time.monotonic(); out = backend.forward(model, inputs, props)
    assert all(torch.isfinite(out[k]).all() for k in props)
    default_forward = dict(config=model.config, seconds=time.monotonic()-start,
                           output_shapes={k:list(out[k].shape) for k in props})
    with args.legacy_checkpoint.open('rb') as handle:
        checkpoint_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    try:
        load_external_checkpoint(args.legacy_checkpoint, device='cpu')
    except ValueError as exc:
        if 'Incompatible QMMM architecture' not in str(exc): raise
        rejection = str(exc)
    else:
        raise AssertionError('Legacy checkpoint was not rejected')
    report = dict(status='PASS', torch_version=torch.__version__, device='CPU', threads=1,
                  dataset=str(args.dataset.resolve()), source_indices=args.source_indices,
                  backward_config=trainer.model.config, baseline_mode='residual_only',
                  loss_weights=weights, epochs=0, optimizer_steps=0, model_weights_unchanged=True,
                  data_files_modified=False, rows=rows, default_model_forward=default_forward,
                  legacy_checkpoint_sha256=checkpoint_sha, legacy_rejection=rejection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('PASS: no optimizer steps, no files modified in source dataset/checkpoint', flush=True)


if __name__ == '__main__':
    main()
