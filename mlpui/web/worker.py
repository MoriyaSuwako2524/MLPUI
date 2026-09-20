"""One isolated training process per run; status writes are atomic."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import threading
from uuid import uuid4


def read_json(path):
    for attempt in range(50):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.02)


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    try:
        # Windows cannot replace a file while a polling reader holds it open.
        for attempt in range(50):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(.02)
    finally:
        temporary.unlink(missing_ok=True)


def run(folder):
    folder = Path(folder)
    # Prevent overlapping workers after a server crash/restart, including the
    # brief interval before the old worker detects its parent has disappeared.
    training_lock = (folder.parent / ".training.lock").open("a+b")
    if training_lock.tell() == 0:
        training_lock.write(b"0")
        training_lock.flush()
    training_lock.seek(0)

    def watch_parent():
        import psutil
        while True:
            time.sleep(1)
            try:
                parent = psutil.Process(int(os.environ["MLPUI_PARENT_PID"]))
                if parent.create_time() != float(os.environ["MLPUI_PARENT_STARTED"]):
                    os._exit(1)
            except (psutil.Error, KeyError):
                os._exit(1)

    if "MLPUI_PARENT_PID" in os.environ:
        threading.Thread(target=watch_parent, daemon=True).start()
    settings = read_json(folder / "config.json")
    state = read_json(folder / "status.json")
    trainer = None
    last_write = 0.

    def publish(**values):
        state.update(values, updated=time.time())
        write_json(folder / "status.json", state)

    def progress(event):
        nonlocal last_write
        now = time.monotonic()
        if event["phase"] in {"epoch_end", "test", "validation"} or now - last_write > .5:
            publish(**event)
            last_write = now
        if event["phase"] == "epoch_end":
            print(json.dumps(event["history"][-1]), flush=True)

    try:
        for attempt in range(50):
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(training_lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if attempt == 49:
                    raise RuntimeError("Another training worker still owns this workspace")
                time.sleep(.1)
        publish(status="running", phase="loading")
        import torch
        from mlpui.training import Trainer, TrainingConfig, TrainingStopped
        from mlpui.web.config import dataset
        options = dict(settings.get("training", {}))
        options["dtype"] = getattr(torch, options.get("dtype", "float32"))
        trainer = Trainer(settings["family"], settings["model_config"], TrainingConfig(**options),
                          checkpoint=settings.get("checkpoint"))
        if settings.get("task_type") == "evaluation":
            data = dataset(settings["evaluation"])
            print(f"Evaluating {len(data)} structures on {trainer.config.device}", flush=True)
            try:
                result = trainer.evaluate(data, on_progress=progress,
                                          should_stop=lambda: (folder / "stop").exists())
            except TrainingStopped:
                publish(status="stopped", phase="stopped")
                return
            write_json(folder / "evaluation.json", result)
            print(json.dumps(result), flush=True)
            publish(status="completed", phase="completed", completed=len(data), total=len(data),
                    evaluation=result)
            return
        train = dataset(settings["train"])
        valid = dataset(settings["validation"]) if settings.get("validation") else None
        test = dataset(settings["test"]) if settings.get("test") else None
        print(f"Training {len(train)} structures on {trainer.config.device}", flush=True)
        try:
            trainer.fit(train, valid, test_data=test, output_dir=folder, on_progress=progress,
                        should_stop=lambda: (folder / "stop").exists())
        except TrainingStopped:
            trainer.save(folder / "model.pt")
            publish(status="stopped", phase="stopped", history=trainer.history,
                    checkpoint=str(folder / "model.pt"))
            return
        publish(status="completed", phase="completed", history=trainer.history,
                checkpoint=str(folder / "model.pt"))
    except Exception as exc:
        traceback.print_exc()
        publish(status="failed", phase="failed", error=str(exc))
        raise
    finally:
        training_lock.close()


if __name__ == "__main__":
    run(sys.argv[1])
