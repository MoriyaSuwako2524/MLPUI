"""MLPUI server entry point."""
import argparse
import asyncio
import webbrowser
from aiohttp import web


async def run(host: str, port: int, open_browser: bool) -> None:
    from app.server import create_app
    app = await create_app()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    url = f"http://{host}:{port}"
    print(f"MLPUI running at {url}")
    if open_browser:
        webbrowser.open(url)

    await asyncio.Event().wait()   # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPUI node-graph server")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int, default=8188)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    asyncio.run(run(args.host, args.port, not args.no_browser))
