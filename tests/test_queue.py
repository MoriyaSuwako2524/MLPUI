import copy
from types import SimpleNamespace

from mlpui.web.server import JobManager, make_server
from mlpui.web.worker import write_json
from test_web import settings, wait_job


class Process:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code

    @property
    def returncode(self):
        return self.code


def fake_manager(tmp_path, monkeypatch, count):
    manager = JobManager(tmp_path / 'runs')
    monkeypatch.setattr(manager, 'enable_scheduler', lambda: None)
    monkeypatch.setattr(manager, 'gpu_inventory', lambda: [dict(device=f'cuda:{i}', idle=True) for i in range(count)])
    monkeypatch.setattr('mlpui.web.server.subprocess.Popen', lambda *a, **kw: Process())
    return manager


def test_gpu_parallel_fifo_and_exit_reservation(tmp_path, monkeypatch):
    manager = fake_manager(tmp_path, monkeypatch, 2)
    config = settings(tmp_path)
    config['training']['device'] = 'cuda'
    jobs = [manager.start(config) for _ in range(4)]
    assert [j['status'] for j in jobs] == ['starting', 'starting', 'queued', 'queued']
    assert {j['assigned_device'] for j in jobs[:2]} == {'cuda:0', 'cuda:1'}
    first = manager.get(jobs[0]['id'])
    first['status'] = 'completed'
    write_json(manager.folder(first['id']) / 'status.json', first)
    manager.tick()
    assert manager.get(jobs[2]['id'])['status'] == 'queued'
    manager.processes[first['id']].code = 0
    manager.tick()
    assert manager.get(jobs[2]['id'])['assigned_device'] == 'cuda:0'
    assert manager.get(jobs[3]['id'])['status'] == 'queued'
    assert manager.stop(jobs[3]['id'])['status'] == 'cancelled'


def test_busy_gpu_and_cpu_independent_queue(tmp_path, monkeypatch):
    manager = fake_manager(tmp_path, monkeypatch, 1)
    monkeypatch.setattr(manager, 'gpu_inventory', lambda: [dict(device='cuda:0', idle=False)])
    config = settings(tmp_path)
    gpu = copy.deepcopy(config)
    gpu['training']['device'] = 'cuda'
    queued = manager.start(gpu)
    assert queued['status'] == 'queued'
    first = manager.start(config)
    second = manager.start(config)
    assert first['assigned_device'] == 'cpu'
    assert second['status'] == 'queued'
    monkeypatch.setattr(manager, 'gpu_inventory', lambda: [dict(device='cuda:0', idle=True)])
    manager.tick()
    assert manager.get(queued['id'])['assigned_device'] == 'cuda:0'
    manager.processes[first['id']].code = 1
    manager.tick()
    assert manager.get(first['id'])['status'] == 'failed'
    assert manager.get(second['id'])['status'] == 'starting'


def test_cpu_queue_runs_without_browser_polling(tmp_path):
    manager = JobManager(tmp_path / 'runs')
    try:
        config = settings(tmp_path, epochs=1)
        first = manager.start(config)
        second = manager.start(config)
        assert second['status'] == 'queued'
        result = wait_job(manager, second['id'])
        assert result['status'] == 'completed', result
        assert manager.get(first['id'])['status'] == 'completed'
    finally:
        manager.close()


def test_queue_survives_shutdown_and_restart(tmp_path, monkeypatch):
    manager = fake_manager(tmp_path, monkeypatch, 0)
    config = settings(tmp_path, epochs=1)
    config['training']['device'] = 'cuda'
    queued = manager.start(config)
    manager.close()
    assert manager.get(queued['id'])['status'] == 'queued'
    # Simulate a CPU queue persisted across a server restart.
    state = manager.get(queued['id'])
    state['requested_device'] = 'cpu'
    write_json(manager.folder(state['id']) / 'status.json', state)
    monkeypatch.undo()
    server = make_server(manager.root, 0)
    try:
        assert wait_job(server.manager, state['id'])['status'] == 'completed'
    finally:
        server.manager.close()
        server.server_close()
        server.lease.close()


def test_probe_failure_keeps_queue_and_cpu_usable(tmp_path, monkeypatch):
    manager = fake_manager(tmp_path, monkeypatch, 1)
    def broken():
        raise RuntimeError('driver unavailable')
    monkeypatch.setattr(manager, 'gpu_inventory', broken)
    config = settings(tmp_path)
    config['training']['device'] = 'cuda'
    assert manager.start(config)['status'] == 'queued'
    config['training']['device'] = 'cpu'
    assert manager.start(config)['assigned_device'] == 'cpu'


def test_gpu_probe_only_uses_visible_devices_and_skips_external_work(monkeypatch):
    import torch
    from mlpui.web.resources import gpu_inventory
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda i: (99, 100))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda i: SimpleNamespace(name='GPU', uuid=f'GPU-visible{i}'))
    monkeypatch.setattr('mlpui.web.resources.subprocess.run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='GPU-visible0, 999999999\nGPU-outside-allocation, 999999998'))
    result = gpu_inventory()
    assert result == [dict(device='cuda:0', name='GPU', idle=False), dict(device='cuda:1', name='GPU', idle=True)]
