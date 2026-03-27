import os
from aiohttp import web

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_web  = os.path.join(_root, "web")


async def create_app() -> web.Application:
    from app.builtin_nodes import load_all_nodes
    load_all_nodes()

    app = web.Application()
    app.router.add_get("/",                handle_index)
    app.router.add_get("/object_info",     handle_object_info)
    app.router.add_post("/prompt",         handle_prompt)
    app.router.add_get("/models/{folder}", handle_model_list)
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


async def handle_model_list(request: web.Request) -> web.Response:
    import app.folder_paths as fp
    files = fp.get_filename_list(request.match_info["folder"])
    return web.json_response(files)
