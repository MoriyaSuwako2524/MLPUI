import json
import threading
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest

from mlpui.web.config import normalize, inspect_data
from mlpui.web.prediction import run_prediction
from mlpui.training import TrainingStopped


def settings(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    np.save(data / 'z.npy', [1, 1, 8])
    np.save(data / 'pos.npy', np.ones((3, 3)))
    np.save(data / 'offsets.npy', [0, 2, 3])
    np.save(data / 'charge.npy', [0, -1])
    np.save(data / 'spin.npy', [1, 2])
    checkpoint = tmp_path / 'uma.pt'
    checkpoint.write_bytes(b'test checkpoint')
    return dict(name='Prediction', family='uma', task_type='prediction',
                checkpoint=str(checkpoint), model_config={'task': 'omol', 'charge': 0, 'spin': 1},
                prediction={'directory': str(data), 'length_scale': 2.}, training={'device': 'cpu'})


def test_prediction_validation_without_labels(tmp_path):
    value = normalize(settings(tmp_path))
    assert inspect_data(value)['prediction']['samples'] == 2
    for patch in ({'checkpoint': ''}, {'family': 'newtonnet'},
                  {'training': {'dtype': 'float64'}}, {'model_config': {'task': 'invalid'}},
                  {'model_config': {'charge': .5}}, {'train': {}}):
        with pytest.raises(ValueError):
            normalize({**value, **patch})
    np.save(tmp_path / 'data' / 'charge.npy', [0, .5])
    with pytest.raises(ValueError, match='integer'):
        normalize(value)


def test_prediction_outputs_and_metadata(tmp_path, monkeypatch):
    value = normalize(settings(tmp_path))
    seen = []
    class Calculator:
        def calculate(self, atoms, properties):
            seen.append(atoms.copy())
            self.results = {'energy': len(atoms) + atoms.info['charge'],
                            'forces': np.ones((len(atoms), 3)) * atoms.info['spin']}
    monkeypatch.setattr('mlpui.web.prediction.build_calculator', lambda *args: Calculator())
    events = []
    result = run_prediction(value, tmp_path, 'cpu', events.append, lambda: False)
    assert [a.info['charge'] for a in seen] == [0, -1]
    assert [a.info['spin'] for a in seen] == [1, 2]
    np.testing.assert_array_equal(seen[0].positions, np.full((2, 3), 2.))
    np.testing.assert_array_equal(np.load(tmp_path / 'offsets.npy'), [0, 2, 3])
    np.testing.assert_array_equal(np.load(tmp_path / 'energy.npy'), [2, 0])
    np.testing.assert_array_equal(np.load(tmp_path / 'forces.npy')[:, 0], [1, 1, 2])
    assert result['units']['energy'] == 'eV'
    assert events[-1]['completed'] == 2
    with pytest.raises(TrainingStopped):
        run_prediction(value, tmp_path, 'cpu', events.append, lambda: True)


def test_official_calculator_adapter(monkeypatch):
    import sys
    from mlpui.web.prediction import build_calculator
    calls = []
    predictor = object()
    def load(**kwargs):
        calls.append(kwargs)
        return predictor
    monkeypatch.setitem(sys.modules, 'fairchem', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'fairchem.core', SimpleNamespace(
        FAIRChemCalculator=lambda model, task_name: (model, task_name)))
    monkeypatch.setitem(sys.modules, 'fairchem.core.units', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'fairchem.core.units.mlip_unit', SimpleNamespace(load_predict_unit=load))
    assert build_calculator('/local/uma.pt', 'omol', 'cuda:1') == (predictor, 'omol')
    assert calls == [{'path': '/local/uma.pt', 'device': 'cuda:1', 'inference_settings': 'default'}]


def test_worker_and_downloads(tmp_path, monkeypatch):
    from mlpui.web.server import make_server
    from mlpui.web.worker import run
    server = make_server(tmp_path / 'runs', 0)
    monkeypatch.setattr(server.manager, 'enable_scheduler', lambda: None)
    monkeypatch.setenv('MLPUI_DEVICE', 'cpu')
    monkeypatch.delenv('MLPUI_PARENT_PID', raising=False)
    monkeypatch.setattr('mlpui.web.prediction.build_calculator', lambda *args: SimpleNamespace(
        calculate=lambda atoms, properties: None,
        results={'energy': 1., 'forces': np.zeros((2, 3))}))
    value = settings(tmp_path)
    # Constant-size data for this worker fixture.
    np.save(tmp_path / 'data' / 'z.npy', [1, 1])
    np.save(tmp_path / 'data' / 'pos.npy', np.zeros((1, 2, 3)))
    value['prediction']['files'] = {'z': 'z.npy', 'pos': 'pos.npy'}
    job = server.manager.start(value)
    folder = server.manager.folder(job['id'])
    run(folder)
    assert server.manager.get(job['id'])['status'] == 'completed'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}/api/jobs/{job["id"]}'
        with urlopen(base + '/prediction') as response:
            assert json.load(response)['samples'] == 1
        with urlopen(base + '/energy.npy') as response:
            assert response.read().startswith(b'\x93NUMPY')
    finally:
        server.shutdown()
        thread.join()
        server.manager.close()
        server.server_close()
        server.lease.close()
