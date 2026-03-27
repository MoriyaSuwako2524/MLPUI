import os
from aiohttp import web

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_web  = os.path.join(_root, "web")


@web.middleware
async def no_cache(request, handler):
    """Disable caching for all responses so JS/CSS changes are always picked up."""
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    return response


_uploads  = os.path.join(_root, "uploads")
_datasets = os.path.join(_root, "dataset")


async def create_app() -> web.Application:
    from app.builtin_nodes import load_all_nodes
    load_all_nodes()

    os.makedirs(_uploads,  exist_ok=True)
    os.makedirs(_datasets, exist_ok=True)

    app = web.Application(
        middlewares=[no_cache],
        client_max_size=256 * 1024 * 1024,
    )
    app.router.add_get("/",                handle_index)
    app.router.add_get("/object_info",     handle_object_info)
    app.router.add_post("/prompt",         handle_prompt)
    app.router.add_post("/upload",         handle_upload)
    app.router.add_get("/xyz_files",           handle_xyz_files)
    app.router.add_get("/history",             handle_history)
    app.router.add_get("/history/{run_id}",    handle_history_run)
    app.router.add_get("/models/{folder}", handle_model_list)
    # Dataset endpoints
    app.router.add_get("/datasets",                    handle_dataset_list)
    app.router.add_get("/datasets/{name}",             handle_dataset_info)
    app.router.add_get("/datasets/{name}/frames",      handle_dataset_frames)
    app.router.add_get("/datasets/{name}/frame/{idx}", handle_dataset_frame)
    app.router.add_post("/datasets/upload",            handle_dataset_upload)
    app.router.add_delete("/datasets/{name}",          handle_dataset_delete)
    app.router.add_static("/web",          _web, show_index=False)
    return app


async def handle_index(request: web.Request) -> web.Response:
    return web.FileResponse(os.path.join(_web, "index.html"))


async def handle_object_info(request: web.Request) -> web.Response:
    from app.nodes import get_object_info
    return web.json_response(get_object_info())


async def handle_prompt(request: web.Request) -> web.Response:
    from app.execution import execute_prompt
    body = await request.json()
    prompt = body.get("prompt", {})
    try:
        result = execute_prompt(prompt)
        return web.json_response(result)
    except Exception as exc:
        import traceback
        return web.json_response(
            {"status": "error", "error": str(exc),
             "traceback": traceback.format_exc()},
            status=500,
        )


async def handle_upload(request: web.Request) -> web.Response:
    try:
        data = await request.post()
        file_field = data.get("file")
        if not file_field or not hasattr(file_field, "filename"):
            return web.json_response({"error": "no file field"}, status=400)

        filename = os.path.basename(file_field.filename)
        dest = os.path.join(_uploads, filename)
        content = file_field.file.read()
        with open(dest, "wb") as f:
            f.write(content)

        return web.json_response({"path": dest, "filename": filename})
    except Exception as exc:
        import traceback
        return web.json_response({"error": str(exc), "traceback": traceback.format_exc()}, status=500)


async def handle_xyz_files(request: web.Request) -> web.Response:
    import app.folder_paths as fp
    return web.json_response(fp.get_xyz_list())


async def handle_history(request: web.Request) -> web.Response:
    import json
    from app.execution import _output_dir
    if not os.path.isdir(_output_dir):
        return web.json_response([])
    runs = []
    for fname in sorted(os.listdir(_output_dir), reverse=True)[:100]:
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_output_dir, fname), encoding="utf-8") as f:
                d = json.load(f)
            # Build a short label from the first STRING output of any output node
            label = ""
            for node in d.get("result", {}).get("ui", {}).values():
                for out in node.get("outputs", []):
                    if out.get("type") == "STRING" and out.get("value"):
                        label = out["value"].splitlines()[0][:60]
                        break
                if label:
                    break
            runs.append({
                "run_id":    d["run_id"],
                "timestamp": d["timestamp"],
                "label":     label,
            })
        except Exception:
            pass
    return web.json_response(runs)


async def handle_history_run(request: web.Request) -> web.Response:
    import json
    from app.execution import _output_dir
    run_id = request.match_info["run_id"]
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        return web.json_response({"error": "invalid run_id"}, status=400)
    path = os.path.join(_output_dir, f"{run_id}.json")
    if not os.path.isfile(path):
        return web.json_response({"error": "not found"}, status=404)
    with open(path, encoding="utf-8") as f:
        return web.json_response(json.load(f))


async def handle_model_list(request: web.Request) -> web.Response:
    import app.folder_paths as fp
    files = fp.get_filename_list(request.match_info["folder"])
    return web.json_response(files)


# ── Dataset handlers ──────────────────────────────────────────────────────────

def _get_dm():
    from mlpui.dataset import DatasetManager
    return DatasetManager(_datasets)


async def handle_dataset_list(request: web.Request) -> web.Response:
    try:
        return web.json_response(_get_dm().list_datasets())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def handle_dataset_info(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    try:
        return web.json_response(_get_dm().get_info(name))
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def handle_dataset_frames(request: web.Request) -> web.Response:
    name  = request.match_info["name"]
    start = int(request.rel_url.query.get("start", 0))
    count = int(request.rel_url.query.get("count", 50))
    count = min(count, 200)  # cap per request
    try:
        frames = _get_dm().get_frames(name, start, count)
        return web.json_response(frames)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        import traceback
        return web.json_response(
            {"error": str(exc), "traceback": traceback.format_exc()}, status=500)


async def handle_dataset_frame(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    try:
        idx = int(request.match_info["idx"])
    except ValueError:
        return web.json_response({"error": "invalid index"}, status=400)
    try:
        return web.json_response(_get_dm().get_frame(name, idx))
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except IndexError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        import traceback
        return web.json_response(
            {"error": str(exc), "traceback": traceback.format_exc()}, status=500)


async def handle_dataset_upload(request: web.Request) -> web.Response:
    try:
        data = await request.post()
        file_field = data.get("file")
        if not file_field or not hasattr(file_field, "filename"):
            return web.json_response({"error": "no file field"}, status=400)
        filename = os.path.basename(file_field.filename)
        content  = file_field.file.read()
        name = _get_dm().upload(filename, content)
        return web.json_response({"name": name})
    except Exception as exc:
        import traceback
        return web.json_response(
            {"error": str(exc), "traceback": traceback.format_exc()}, status=500)


async def handle_dataset_delete(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    try:
        _get_dm().delete(name)
        return web.json_response({"deleted": name})
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
