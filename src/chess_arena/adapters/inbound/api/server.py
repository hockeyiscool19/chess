"""Run the app with uvicorn."""

import uvicorn

from chess_arena.adapters.inbound.api.app import create_app
from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.adapters.inbound.settings import ArenaSettings


def serve(settings: ArenaSettings, host: str | None = None, port: int | None = None) -> int:
    """Serve the web app and API until interrupted; return exit code 0."""
    arena = Arena(settings)
    bind_host = host or settings.host
    bind_port = port or settings.port
    print(f"Chess Arena on http://{bind_host}:{bind_port}  (home {arena.home})", flush=True)
    uvicorn.run(create_app(arena), host=bind_host, port=bind_port, log_level="info")
    return 0
