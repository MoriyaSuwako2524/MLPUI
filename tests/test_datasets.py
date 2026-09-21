import io
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest

from mlpui.web.datasets import DatasetManager
from mlpui.web.config import dataset, normalize, inspect_data, presets
from mlpui.web.server import make_server


def source_data(path, ragged=False):
    path.mkdir()
    n = 10
    sizes = np.arange(n) % 3 + 1 if ragged else np.full(n, 3)
    z = np.ones(sizes.sum() if ragged else 3, dtype=int)
    pos = np.arange(sizes.sum() * 3, dtype=float).reshape((-1, 3) if ragged else (n, 3, 3))
    arrays = dict(z=z, pos=pos, forces=pos * .01, energy=np.arange(n).reshape(n, 1),
                  charges=np.ones((sizes.sum(), 1) if ragged else (n, 3, 1)) * .25,
                  charge=(sizes * .25).reshape(n, 1), spin=np.zeros(n),
                  cell=np.tile(np.eye(3) * 50, (n, 1, 1)), pbc=np.zeros((n, 3), dtype=bool),
                  dipole=np.zeros((n, 3)), stress=np.zeros((n, 3, 3)))
    if ragged:
        arrays['offsets'] = np.concatenate(([0], np.cumsum(sizes)))
    for key, value in arrays.items():
        np.save(path / f'{key}.npy', value)
    return path


@pytest.mark.parametrize('ragged', [False, True])
def test_split_reproducible_complete_aligned_and_units(tmp_path, ragged):
    manager = DatasetManager(tmp_path / 'catalog')
    path = source_data(tmp_path / 'source', ragged)
    original = {p.name: p.read_bytes() for p in path.iterdir()}
    record = manager.create('source', dict(directory=str(path), gradients=True, energy_scale=2., length_scale=3.))
    options = dict(name='split', ratios=[6, 2, 2], seed=42)
    result = manager.split(record['id'], options)
    repeated = manager.split(record['id'], options)
    assert result['manifest']['counts'] == [6, 2, 2]
    source = dataset(record['spec'])
    indices = []
    for child, again in zip(result['datasets'], repeated['datasets']):
        split_path = manager.root / ('split-' + child['split_id'])
        selected = np.load(split_path / (child['split_role'] + '_indices.npy'))
        indices.extend(selected.tolist())
        data = dataset(child['spec'])
        data_again = dataset(again['spec'])
        for row, index in enumerate(selected):
            assert set(data[row]) == set(source[int(index)])
            for key, value in data[row].items():
                np.testing.assert_allclose(value.reshape(-1), source[int(index)][key].reshape(-1))
                np.testing.assert_array_equal(value, data_again[row][key])
    assert sorted(indices) == list(range(10))
    assert len(set(indices)) == 10
    assert original == {p.name: p.read_bytes() for p in path.iterdir()}
    specs = {d['split_role']: d['spec'] for d in result['datasets']}
    config = normalize(dict(name='test', family='newtonnet', model_config=presets()['newtonnet'],
                            training={}, **specs))
    assert inspect_data(config)['train']['samples'] == 6
    assert len(DatasetManager(manager.root).list()) == 7


def test_custom_shards_and_ordered_split(tmp_path):
    path = source_data(tmp_path / 'source')
    np.save(path / 'coord_a.npy', np.zeros((2, 3, 3)))
    np.save(path / 'coord_b.npy', np.ones((3, 3, 3)))
    np.save(path / 'e_a.npy', np.arange(2))
    np.save(path / 'e_b.npy', np.arange(2, 5))
    manager = DatasetManager(tmp_path / 'catalog')
    record = manager.create('groups', dict(directory=str(path), shards=['a', 'b'],
        files=dict(z='z.npy', pos='coord_{shard}.npy', energy='e_{shard}.npy')))
    result = manager.split(record['id'], dict(name='ordered', ratios=[3, 0, 2], method='ordered'))
    assert result['manifest']['counts'] == [3, 0, 2]
    assert [float(dataset(d['spec'])[0]['energy']) for d in result['datasets']] == [0., 3.]


def test_catalog_rename_archive_and_validation(tmp_path):
    manager = DatasetManager(tmp_path / 'catalog')
    record = manager.create('original', dict(directory=str(source_data(tmp_path / 'source'))))
    manager.update(record['id'], dict(name='renamed', archived=True))
    assert manager.get(record['id'])['name'] == 'renamed'
    with pytest.raises(ValueError):
        manager.split(record['id'], dict(name='x'))
    manager.update(record['id'], dict(archived=False))
    for options in [dict(ratios=[1, -1, 1]), dict(ratios=[1, 1, 1000]), dict(seed=-1), dict(method='unknown')]:
        with pytest.raises(ValueError):
            manager.split(record['id'], dict(name='invalid', **options))
    with pytest.raises(ValueError):
        manager.get('../source')


def test_stream_upload_and_finalize(tmp_path):
    manager = DatasetManager(tmp_path / 'catalog')
    record = manager.create('upload')
    source = source_data(tmp_path / 'source', True)
    for path in source.iterdir():
        content = path.read_bytes()
        manager.upload(record['id'], path.name, io.BytesIO(content), len(content))
    with pytest.raises(ValueError):
        manager.upload(record['id'], '../escape.npy', io.BytesIO(b'x'), 1)
    with pytest.raises(ValueError):
        manager.upload(record['id'], 'z.npy', io.BytesIO(b'x'), 1)
    with pytest.raises(ValueError):
        manager.upload(record['id'], 'bad.npy', io.BytesIO(b'junk'), 4)
    with pytest.raises(ValueError):
        manager.upload(record['id'], 'short.npy', io.BytesIO(b'x'), 100)
    objects = io.BytesIO()
    np.save(objects, np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        manager.upload(record['id'], 'objects.npy', io.BytesIO(objects.getvalue()), objects.tell())
    with pytest.raises(ValueError):
        manager.finalize(record['id'], dict(files=dict(z='../z.npy')))
    result = manager.finalize(record['id'], {})
    assert result['status'] == 'ready' and result['summary']['samples'] == 10
    assert not list((manager.folder(record['id']) / 'data').glob('*.part'))
    with pytest.raises(ValueError):
        manager.upload(record['id'], 'extra.npy', io.BytesIO(b'x'), 1)


def test_dataset_http_lifecycle(tmp_path):
    server = make_server(tmp_path / 'runs', 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    def post(path, value):
        with urlopen(Request(base + path, json.dumps(value).encode(), {'Content-Type': 'application/json'})) as response:
            return json.load(response)
    try:
        record = post('/api/datasets', dict(name='uploaded'))
        for path in source_data(tmp_path / 'source').iterdir():
            with urlopen(Request(base + f'/api/datasets/{record["id"]}/files/{path.name}', path.read_bytes(),
                                 {'Content-Type': 'application/octet-stream'})) as response:
                assert response.status == 201
        record = post(f'/api/datasets/{record["id"]}/finalize', {})
        assert record['summary']['samples'] == 10
        result = post(f'/api/datasets/{record["id"]}/split', dict(name='split', ratios=[8, 1, 1], seed=10))
        assert len(result['datasets']) == 3
        post(f'/api/datasets/{record["id"]}/update', dict(archived=True))
        with urlopen(base + '/api/datasets') as response:
            assert len(json.load(response)) == 4
        with pytest.raises(HTTPError):
            urlopen(Request(base + '/api/datasets', b'{}', {'Content-Type': 'application/json', 'Origin': 'https://example.org'}))
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        server.lease.close()


def test_tags_validation_persistence_and_split_inheritance(tmp_path):
    manager = DatasetManager(tmp_path / 'catalog')
    record = manager.create('tagged', dict(directory=str(source_data(tmp_path / 'source'))),
                            tags=[' DFT ', 'dft', '反应'])
    assert record['tags'] == ['DFT', '反应']
    children = manager.split(record['id'], dict(name='split'))['datasets']
    assert all(child['tags'] == ['DFT', '反应'] for child in children)
    manager.update(record['id'], dict(tags=['v2']))
    assert DatasetManager(manager.root).get(record['id'])['tags'] == ['v2']
    for tags in ['bad', [''], ['x' * 41], [3], ['x'] * 21]:
        with pytest.raises(ValueError):
            manager.update(record['id'], dict(tags=tags))
    manager.update(record['id'], dict(tags=[]))
    assert manager.get(record['id'])['tags'] == []


def test_delete_existing_record_keeps_original_files(tmp_path):
    manager = DatasetManager(tmp_path / 'catalog')
    path = source_data(tmp_path / 'source')
    record = manager.create('source', dict(directory=str(path)))
    with pytest.raises(ValueError, match='源文件'):
        manager.delete(record['id'], dict(delete_files=True))
    manager.delete(record['id'], {})
    assert not manager.list()
    assert (path / 'pos.npy').exists()
    assert (manager.folder(record['id']) / 'deleted.json').exists()


def test_delete_managed_files_reference_guards_and_siblings(tmp_path):
    manager = DatasetManager(tmp_path / 'catalog')
    source = manager.create('source', dict(directory=str(source_data(tmp_path / 'source'))))
    children = manager.split(source['id'], dict(name='split'))['datasets']
    child = children[0]
    from pathlib import Path
    path = Path(child['spec']['directory'])
    with pytest.raises(ValueError, match='任务引用'):
        manager.delete(child['id'], dict(delete_files=True), job_specs=[child['spec']])
    # Explicit file mappings outside the referencing directory also count.
    with pytest.raises(ValueError, match='任务引用'):
        manager.delete(child['id'], dict(delete_files=True), job_specs=[dict(directory=str(tmp_path / 'elsewhere'), files={'pos': str(path / 'pos.npy')})])
    alias = manager.create('alias', child['spec'])
    with pytest.raises(ValueError, match='其他数据集'):
        manager.delete(child['id'], dict(delete_files=True))
    manager.delete(alias['id'], {})
    manager.delete(child['id'], dict(delete_files=True))
    assert not path.exists()
    assert all(Path(d['spec']['directory']).exists() for d in children[1:])
    assert (path.parent / 'split.json').exists()


def test_delete_upload_draft_and_reject_tampered_path(tmp_path):
    manager = DatasetManager(tmp_path / 'catalog')
    draft = manager.create('draft')
    folder = manager.folder(draft['id']) / 'data'
    (folder / 'leftover.part').write_bytes(b'partial')
    manager.delete(draft['id'], dict(delete_files=True))
    assert not folder.exists()
    source = manager.create('source', dict(directory=str(source_data(tmp_path / 'source'))))
    child = manager.split(source['id'], dict(name='split'))['datasets'][0]
    from mlpui.web.worker import write_json
    child['spec']['directory'] = str(tmp_path)
    write_json(manager.folder(child['id']) / 'dataset.json', child)
    with pytest.raises(ValueError, match='owned directory'):
        manager.delete(child['id'], dict(delete_files=True))
