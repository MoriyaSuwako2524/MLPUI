import io
import json
import threading
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from mlpui.training import Trainer, TrainingConfig
from mlpui.web.config import normalize
from mlpui.web.server import make_server
from mlpui.calculator import CalculatorBuilder
from test_training import write_data
from test_web import settings, wait_job


def controlled_trainer(tmp_path, sequence, **options):
    trainer = Trainer.__new__(Trainer)
    # This fixture bypasses initialization to isolate the early-stopping loop.
    from mlpui.backends import get_backend
    trainer.backend = get_backend("newtonnet")
    trainer.family = trainer.backend.name
    trainer.config = TrainingConfig(epochs=10, loss_weights={'energy': 1.}, early_stopping=True,
                                   early_stopping_patience=2, **options)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01)
    trainer.model_config = {}
    trainer.history = []
    train, valid, test = [write_data(tmp_path / name) for name in ('train', 'valid', 'test')]
    for path, marker in ((train, 0), (valid, 1), (test, 2)):
        np.save(path / 'energy.npy', np.full(2, marker))
    def loss(sample):
        if sample['energy'] == 0:
            value = trainer.model.weight.square().sum()
        else:
            value = torch.tensor(sequence[len(trainer.history)] if sample['energy'] == 1 else 1000. - len(trainer.history))
        return value, {'energy': value}
    trainer._loss = loss
    return trainer, train, valid, test


def test_stops_on_validation_and_preserves_best_separately(tmp_path):
    trainer, train, valid, test = controlled_trainer(tmp_path, [5., 4., 4.1, 4.2])
    trainer.fit(train, valid, test_data=test, output_dir=tmp_path / 'run')
    assert len(trainer.history) == 4
    assert trainer.early_stopping['stopped']
    best = torch.load(tmp_path / 'run/best.pt', weights_only=True)
    final = torch.load(tmp_path / 'run/model.pt', weights_only=True)
    assert best['epoch'] == 2 and final['epoch'] == 4
    assert best['early_stopping']['best_value'] == 4.
    assert not torch.equal(best['state_dict']['weight'], final['state_dict']['weight'])


def test_min_delta_and_actual_best(tmp_path):
    trainer, train, valid, _ = controlled_trainer(tmp_path, [5., 4.875, 4.75], early_stopping_min_delta=.5)
    trainer.fit(train, valid, output_dir=tmp_path / 'run')
    assert len(trainer.history) == 3
    assert trainer.early_stopping['best_epoch'] == 3
    assert torch.load(tmp_path / 'run/best.pt', weights_only=True)['epoch'] == 3


def test_missing_validation_and_invalid_parameters(tmp_path):
    trainer, train, _, _ = controlled_trainer(tmp_path, [1.])
    with pytest.raises(ValueError, match='validation'):
        trainer.fit(train, output_dir=tmp_path / 'run')
    for kwargs in [dict(early_stopping_patience=0), dict(early_stopping_patience=True),
                   dict(early_stopping_min_delta=-1), dict(early_stopping_min_delta=float('nan')),
                   dict(early_stopping_monitor='test'), dict(early_stopping='yes')]:
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)
    config = settings(tmp_path)
    config['training']['early_stopping'] = True
    with pytest.raises(ValueError, match='验证集'):
        normalize(config)


@pytest.mark.parametrize('family', ['newtonnet', 'torchmdnet'])
def test_real_worker_early_stop_best_download_and_queue(tmp_path, family):
    config = settings(tmp_path, epochs=8)
    if family == 'torchmdnet':
        from test_external_models_integration import torchmd_args
        config.update(family=family, model_config=torchmd_args('tensornet'))
    config['validation'] = dict(directory=str(write_data(tmp_path / 'valid')))
    config['training'].update(early_stopping=True, early_stopping_patience=1,
                              early_stopping_min_delta=1e6, early_stopping_monitor='forces')
    server = make_server(tmp_path / 'runs', 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(Request(base + '/api/jobs', json.dumps(config).encode(), {'Content-Type': 'application/json'})) as response:
            first = json.load(response)
        config['training']['epochs'] = 1
        second = server.manager.start(config)
        result = wait_job(server.manager, first['id'])
        assert result['status'] == 'completed' and result['stop_reason'] == 'early_stopping'
        assert len(result['history']) == 2 and result['best_checkpoint']
        with urlopen(base + f'/api/jobs/{first["id"]}/best') as response:
            saved = torch.load(io.BytesIO(response.read()), weights_only=True)
        assert saved['epoch'] == result['early_stopping']['best_epoch']
        CalculatorBuilder.from_checkpoint(server.manager.folder(first['id']) / 'best.pt', device='cpu').build()
        assert wait_job(server.manager, second['id'])['status'] == 'completed'
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        server.lease.close()
