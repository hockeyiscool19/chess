"""FastAPI application factory. OpenAPI stays enabled at ``/docs`` and ``/openapi.json``."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chess_arena.adapters.inbound.api import errors
from chess_arena.adapters.inbound.api.jobs import BackgroundJobs
from chess_arena.adapters.inbound.api.routes_jobs import jobs_router
from chess_arena.adapters.inbound.api.routes_models import models_router
from chess_arena.adapters.inbound.api.routes_play import play_router
from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.domain.contract import ArenaModel


class Health(ArenaModel):
    """Liveness plus the facts a client needs before queuing work."""

    status: str
    stockfish: str | None
    models: int
    workers: int


def create_app(arena: Arena) -> FastAPI:
    """Return the app bound to ``arena``; the arena's pool closes on shutdown."""
    jobs = BackgroundJobs(arena)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        """Stop background jobs and worker processes when the server exits."""
        yield
        jobs.shutdown()
        arena.close()

    app = FastAPI(
        title="Chess Arena",
        version="0.1.0",
        description="Play bots, run engine ladders, and publish trained models.",
        lifespan=lifespan,
    )
    errors.install(app)
    app.include_router(play_router(arena))
    app.include_router(models_router(arena))
    app.include_router(jobs_router(arena, jobs))

    @app.get("/api/health", tags=["meta"])
    def health() -> Health:
        """Report liveness, Stockfish availability, and pool size."""
        return Health(
            status="ok",
            stockfish=arena.stockfish,
            models=len(arena.models.cards()),
            workers=arena.settings.workers,
        )

    static = resources.files("chess_arena.adapters.inbound.api") / "static"
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page web app."""
        return FileResponse(str(static / "index.html"))

    return app
