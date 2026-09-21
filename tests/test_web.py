import importlib
import copy
import json
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest
import torch

from mlpui.web.config import normalize, inspect_data, presets
from mlpui.web.server import make_server, JobManager
from mlpui.web.worker import write_json
from mlpui.training import Trainer, TrainingConfig, TrainingStopped
from test_training import write_data


def settings(tmp_path, epochs=2):
    data = write_data(tmp_path / "data")
    return {"name": "Test run", "family": "newtonnet",
            "model_config": dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                                 output_properties=["energy", "gradient_force"]),
            "training": {"epochs": epochs, "batch_size": 1},
            "train": {"directory": str(data)}}


def wait_job(manager, job_id, terminal=True):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = manager.get(job_id)
        if state["status"] not in {"queued", "starting", "running", "stopping"}:
            return state
        if not terminal and (state.get("phase") == "training" or state.get("history")):
            return state
        time.sleep(.1)
    pytest.fail("Worker did not reach the expected state")


def test_validation_and_preview(tmp_path):
    value = settings(tmp_path)
    assert inspect_data(normalize(value))["train"]["samples"] == 2
    value["validation"] = copy.deepcopy(value["train"])
    with pytest.raises(ValueError, match="separate"):
        normalize(value)
    value.pop("validation")
    value["training"]["epochs"] = 0
    with pytest.raises(ValueError):
        normalize(value)
    value["training"]["epochs"] = 2
    value["train"]["files"] = {"z": "z.npy", "pos": "pos.npy"}
    with pytest.raises(ValueError, match="missing labels"):
        inspect_data(normalize(value))


def test_status_write_retries_windows_sharing_violation(tmp_path, monkeypatch):
    import os
    original = os.replace
    attempts = []

    def sharing_violation(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("Reader temporarily holds status.json")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", sharing_violation)
    write_json(tmp_path / "status.json", {"status": "running"})
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.integration
def test_real_job_stop_failure_and_reload(tmp_path):
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet")
    manager = JobManager(tmp_path / "runs")
    try:
        config = settings(tmp_path)
        job = manager.start(config)
        result = wait_job(manager, job["id"])
        assert result["status"] == "completed", result
        assert len(result["history"]) == 2
        assert torch.load(result["checkpoint"], weights_only=True)["history"]
        assert JobManager(tmp_path / "runs").list()[0]["status"] == "completed"
        config["training"]["epochs"] = 10000
        job = manager.start(config)
        queued = manager.start(config)
        assert queued["status"] == "queued"
        assert manager.stop(queued["id"])["status"] == "cancelled"
        wait_job(manager, job["id"], terminal=False)
        manager.stop(job["id"])
        result = wait_job(manager, job["id"])
        assert result["status"] == "stopped", result
        assert torch.load(result["checkpoint"], weights_only=True)["state_dict"]
        config["model_config"]["output_properties"] = ["energy"]
        job = manager.start(config)
        result = wait_job(manager, job["id"])
        assert result["status"] == "failed"
        assert "forces" in result["error"]
    finally:
        manager.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_http_and_origin_guard(tmp_path, host):
    server = make_server(tmp_path / "runs", 0, host=host)
    assert server.server_address[0] == host
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert b"TRAINING STUDIO" in response.read()
        with urlopen(base + "/app.js") as response:
            assert b"renderDetail" in response.read()
        request = Request(base + "/api/preview", json.dumps(settings(tmp_path)).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert json.load(response)["train"]["samples"] == 2
        request.add_header("Origin", "https://unrelated.example")
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 403
        with pytest.raises(HTTPError):
            urlopen(base + "/../pyproject.toml")
        with pytest.raises(OSError):
            make_server(tmp_path / "runs", 0)
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        server.lease.close()


@pytest.mark.integration
@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_web_presets_train(tmp_path, family):
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet" if family == "newtonnet" else "mlpui.models.torchmdnet.models.model")
    config = presets()[family]
    if family == "newtonnet":
        config.update(n_features=8, n_basis=4, n_interactions=1)
    else:
        config.update(embedding_dimension=8, num_rbf=4, num_layers=1)
    trainer = Trainer(family, config, TrainingConfig(epochs=1))
    assert trainer.fit(write_data(tmp_path / "data"))[0]["train"]["energy"] >= 0


@pytest.mark.integration
def test_http_training_and_download(tmp_path):
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet")
    server = make_server(tmp_path / "runs", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        config = settings(tmp_path, epochs=3)
        config["training"].update(save_interval=1, max_checkpoints=2, test_interval=2)
        config["test"] = {"directory": str(write_data(tmp_path / "test"))}
        request = Request(base + "/api/jobs", json.dumps(config).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert response.status == 201
            job = json.load(response)
        state = wait_job(server.manager, job["id"])
        assert state["status"] == "completed", state
        assert [r["epoch"] for r in state["history"] if "test" in r] == [2]
        assert state["checkpoints"] == ["epoch_000003.pt", "epoch_000002.pt"]
        with urlopen(base + f'/api/jobs/{job["id"]}/checkpoints/epoch_000002.pt') as response:
            assert response.read()[:2] == b"PK"
        with pytest.raises(HTTPError):
            urlopen(base + f'/api/jobs/{job["id"]}/checkpoints/epoch_000001.pt')
        with urlopen(base + f'/api/jobs/{job["id"]}/model') as response:
            assert response.headers["Content-Disposition"].endswith('"model.pt"')
            assert response.read()[:2] == b"PK"
        with urlopen(base + f'/api/jobs/{job["id"]}/log') as response:
            assert "epoch" in json.load(response)["text"]
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        server.lease.close()


def test_evaluation_task_and_result_endpoint(tmp_path):
    config = settings(tmp_path)
    config["task_type"] = "evaluation"
    config["evaluation"] = config.pop("train")
    with pytest.raises(ValueError, match="checkpoint"):
        normalize(config)
    trainer = Trainer(config["family"], config["model_config"])
    checkpoint = trainer.save(tmp_path / "source.pt")
    before = checkpoint.read_bytes()
    config["checkpoint"] = str(checkpoint)
    assert inspect_data(normalize(config))["evaluation"]["samples"] == 2
    server = make_server(tmp_path / "runs", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(Request(base + "/api/jobs", json.dumps(config).encode(),
                             {"Content-Type": "application/json"})) as response:
            job = json.load(response)
        state = wait_job(server.manager, job["id"])
        assert state["status"] == "completed", state
        assert state["task_type"] == "evaluation"
        assert state["evaluation"]["metrics"]["forces"]["count"] == 18
        assert set(state["evaluation"]["plots"]) == {"energy", "forces"}
        for key in ("energy", "forces"):
            for extension, mime in (("png", "image/png"), ("svg", "image/svg+xml")):
                with urlopen(base + f'/api/jobs/{job["id"]}/plots/{key}.{extension}') as response:
                    assert response.headers["Content-Type"] == mime
                    content = response.read()
                    assert content.startswith(b"\x89PNG") if extension == "png" else b"<svg" in content
        with urlopen(base + f'/api/jobs/{job["id"]}/evaluation') as response:
            assert json.load(response) == state["evaluation"]
        assert not (server.manager.folder(job["id"]) / "model.pt").exists()
        assert checkpoint.read_bytes() == before
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        server.lease.close()


def test_test_split_preview_and_overlap(tmp_path):
    config = settings(tmp_path)
    config["test"] = {"directory": str(write_data(tmp_path / "test"))}
    assert inspect_data(normalize(config))["test"]["samples"] == 2
    config["test"] = dict(config["train"], files={"pos": "pos.npy", "z": "z.npy"})
    with pytest.raises(ValueError, match="separate"):
        normalize(config)


def test_crashed_session_is_interrupted(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    write_json(folder / "status.json", {"id": folder.name, "status": "running", "created": 0})
    server = make_server(tmp_path, 0)
    try:
        assert server.manager.get(folder.name)["status"] == "interrupted"
    finally:
        server.server_close()
        server.lease.close()


@pytest.mark.integration
def test_progress_and_cooperative_stop(tmp_path):
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet")
    config = settings(tmp_path)
    trainer = Trainer("newtonnet", config["model_config"], TrainingConfig(epochs=2))
    events = []
    with pytest.raises(TrainingStopped):
        trainer.fit(config["train"]["directory"], on_progress=events.append,
                    should_stop=lambda: bool(events))
    assert events[0]["phase"] == "training"
    assert events[0]["completed"] == 1
