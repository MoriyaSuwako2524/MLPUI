"""Loopback-only dashboard; no Node.js or additional web framework required."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
from uuid import uuid4

from mlpui.web.config import normalize, inspect_data, presets
from mlpui.web.worker import write_json, read_json
from mlpui.web.datasets import DatasetManager

ACTIVE = {"starting", "running", "stopping"}


class JobManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        # Server shutdown owns its workers. Persisted active records from a crash
        # are reconciled by the worker lease, before another task is allowed.

    def folder(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid task ID")
        path = self.root / job_id
        if not path.is_dir():
            raise FileNotFoundError("Task not found")
        return path

    def get(self, job_id):
        with self.lock:
            folder = self.folder(job_id)
            state = read_json(folder / "status.json")
            process = self.processes.get(job_id)
            if state["status"] in ACTIVE and process is not None and process.poll() is not None:
                # The worker may have committed its final status after our read.
                state = read_json(folder / "status.json")
                if state["status"] in ACTIVE:
                    state.update(status="failed", error=f"Worker exited with code {process.returncode}")
                    write_json(folder / "status.json", state)
            state["stop_requested"] = (folder / "stop").exists()
            directory = folder / "checkpoints"
            state["checkpoints"] = sorted(
                [p.name for p in directory.glob("epoch_*.pt")
                 if p.is_file() and re.fullmatch(r"epoch_[0-9]{6,}\.pt", p.name)],
                key=lambda name: int(name[6:-3]), reverse=True)
            return state

    def list(self):
        return sorted([self.get(p.name) for p in self.root.iterdir()
                       if p.is_dir() and re.fullmatch(r"[a-f0-9]{32}", p.name)
                       and (p / "status.json").exists()], key=lambda v: v["created"], reverse=True)

    def start(self, payload):
        settings = normalize(payload)
        summary = inspect_data(settings)
        with self.lock:
            if any(job["status"] in ACTIVE for job in self.list()):
                raise ValueError("已有任务在运行，请等待完成或停止后再启动。")
            job_id = uuid4().hex
            folder = self.root / job_id
            folder.mkdir()
            state = {"id": job_id, "name": settings["name"], "family": settings["family"],
                     "task_type": settings["task_type"],
                     "status": "starting", "created": time.time(), "epoch": 0,
                     "epochs": settings["training"].get("epochs", 10), "history": [],
                     "summary": summary, "directory": str(folder)}
            write_json(folder / "config.json", settings)
            write_json(folder / "status.json", state)
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            import psutil
            env["MLPUI_PARENT_PID"] = str(os.getpid())
            env["MLPUI_PARENT_STARTED"] = str(psutil.Process().create_time())
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
            try:
                with (folder / "train.log").open("wb") as log:
                    self.processes[job_id] = subprocess.Popen(
                        [sys.executable, "-u", "-m", "mlpui.web.worker", str(folder)],
                        stdout=log, stderr=subprocess.STDOUT, env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except Exception as exc:
                state.update(status="failed", error=str(exc))
                write_json(folder / "status.json", state)
                raise
            return state

    def stop(self, job_id):
        with self.lock:
            state = self.get(job_id)
            if state["status"] in ACTIVE:
                (self.folder(job_id) / "stop").touch()
            return self.get(job_id)

    def close(self):
        for job_id, process in self.processes.items():
            if process.poll() is None:
                self.stop(job_id)
        for job_id, process in self.processes.items():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
                folder = self.folder(job_id)
                state = self.get(job_id)
                state.update(status="interrupted", error="WebUI closed before the worker could stop safely")
                write_json(folder / "status.json", state)


def make_server(root, port=8675, host="127.0.0.1"):
    manager = JobManager(root)
    datasets = DatasetManager(manager.root / "datasets")
    # Reject a second server using the same run directory. OS releases lock on
    # crashes; any orphan is terminated by the worker's parent-liveness check.
    lease = (manager.root / ".server.lock").open("a+b")
    if lease.tell() == 0:
        lease.write(b"0")
        lease.flush()
    lease.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for job in manager.list():
        if job["status"] in ACTIVE:
            job.update(status="interrupted", error="Previous WebUI session ended unexpectedly")
            write_json(manager.folder(job["id"]) / "status.json", job)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def check_local(self):
            host = self.headers.get("Host", "")
            if host not in (f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"):
                raise PermissionError("Invalid host")
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                raise PermissionError("Cross-origin requests are not allowed")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise PermissionError("Cross-site requests are not allowed")

        def do_GET(self):
            try:
                self.check_local()
                path = urlparse(self.path).path
                if path == "/api/jobs":
                    return self.respond(manager.list())
                if path == "/api/datasets":
                    return self.respond(datasets.list())
                match = re.fullmatch(r"/api/datasets/([a-f0-9]{32})", path)
                if match:
                    return self.respond(datasets.get(match[1]))
                if path == "/api/presets":
                    import torch
                    return self.respond({"models": presets(), "cuda": torch.cuda.is_available(),
                                         "root": str(manager.root)})
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(log|config|evaluation|model|plots/(?:energy|forces|charges|dipole|stress)\.(?:png|svg)|checkpoints/epoch_[0-9]{6,}\.pt))?", path)
                if match:
                    job_id, action = match.groups()
                    folder = manager.folder(job_id)
                    if action == "log":
                        file = folder / "train.log"
                        text = ""
                        if file.exists():
                            with file.open("rb") as stream:
                                stream.seek(max(0, file.stat().st_size - 32000))
                                text = stream.read().decode("utf-8", errors="replace")
                        return self.respond({"text": text})
                    if action == "config":
                        return self.respond(read_json(folder / "config.json"))
                    if action == "evaluation":
                        return self.respond(read_json(folder / "evaluation.json"))
                    if action and action.startswith("plots/"):
                        file = folder / action
                        content = file.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png" if file.suffix == ".png" else "image/svg+xml")
                        self.send_header("Content-Length", str(len(content)))
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        self.wfile.write(content)
                        return
                    if action == "model" or (action and action.startswith("checkpoints/")):
                        if action == "model":
                            if manager.get(job_id)["status"] not in {"completed", "stopped"}:
                                raise ValueError("Model is not ready")
                            file = folder / "model.pt"
                        else:
                            file = folder / action
                        # Open before sending headers; retention can remove an older
                        # checkpoint while a user selects its download link.
                        with file.open("rb") as stream:
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Disposition", f'attachment; filename="{file.name}"')
                            self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                            self.end_headers()
                            while chunk := stream.read(1024 * 1024):
                                self.wfile.write(chunk)
                        return
                    return self.respond(manager.get(job_id))
                assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                          "/style.css": ("style.css", "text/css"),
                          "/datasets.js": ("datasets.js", "text/javascript")}
                if path not in assets:
                    return self.respond({"error": "Not found"}, 404)
                filename, mime = assets[path]
                content = (Path(__file__).parent / "static" / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(content)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.respond({"error": str(exc)}, 403 if isinstance(exc, PermissionError) else 400)

        def do_POST(self):
            try:
                self.check_local()
                match = re.fullmatch(r"/api/datasets/([a-f0-9]{32})/files/([^/]+)", self.path)
                if match:
                    if self.headers.get("Content-Type") != "application/octet-stream":
                        raise ValueError("Expected application/octet-stream")
                    self.connection.settimeout(120)
                    self.close_connection = True
                    return self.respond(datasets.upload(match[1], unquote(match[2]), self.rfile,
                                                       int(self.headers.get("Content-Length", "0"))), 201)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/datasets":
                    return self.respond(datasets.create(payload.get("name"), payload.get("spec"), payload.get("tags")), 201)
                match = re.fullmatch(r"/api/datasets/([a-f0-9]{32})/(update|finalize|split|inspect|delete)", self.path)
                if match:
                    identifier, action = match.groups()
                    if action == "delete":
                        with manager.lock:
                            specs = []
                            for job in manager.list():
                                config = read_json(manager.folder(job["id"]) / "config.json")
                                specs.extend(config[key] for key in ("train", "validation", "test", "evaluation") if config.get(key))
                            return self.respond(datasets.delete(identifier, payload, job_specs=specs))
                    if action == "inspect":
                        return self.respond(datasets.inspect(identifier))
                    method = {"update": datasets.update, "finalize": datasets.finalize, "split": datasets.split}[action]
                    return self.respond(method(identifier, payload))
                if self.path == "/api/preview":
                    return self.respond(inspect_data(normalize(payload)))
                if self.path == "/api/jobs":
                    with manager.lock:
                        return self.respond(manager.start(payload), 201)
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/stop", self.path)
                if match:
                    return self.respond(manager.stop(match[1]))
                self.respond({"error": "Not found"}, 404)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.respond({"error": str(exc)}, 403 if isinstance(exc, PermissionError) else 400)

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except Exception:
        lease.close()
        raise
    server.manager = manager
    server.datasets = datasets
    server.lease = lease
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8675)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; 0.0.0.0 allows cluster-network access (no authentication)")
    parser.add_argument("--runs-dir", default="runs/web")
    args = parser.parse_args()
    server = make_server(args.runs_dir, args.port, host=args.host)
    print(f"MLPUI listening: {args.host}:{server.server_port}", flush=True)
    print(f"Browser via SSH tunnel: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.manager.close()
        server.server_close()
        server.lease.close()
