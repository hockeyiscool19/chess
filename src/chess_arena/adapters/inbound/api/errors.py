"""Map arena errors onto HTTP status codes."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from chess_arena.domain.errors import (
    ArenaError,
    ConflictError,
    EngineUnavailableError,
    GameOverError,
    IllegalMoveError,
    InvalidRecipeError,
    NotFoundError,
)

_STATUS: tuple[tuple[type[ArenaError], int], ...] = (
    (NotFoundError, 404),
    (ConflictError, 409),
    (GameOverError, 409),
    (IllegalMoveError, 422),
    (InvalidRecipeError, 422),
    (EngineUnavailableError, 503),
)


def status_for(exc: ArenaError) -> int:
    """Return the HTTP status for an arena error (400 when unmapped)."""
    for kind, status in _STATUS:
        if isinstance(exc, kind):
            return status
    return 400


async def _arena_error(request: Request, exc: Exception) -> JSONResponse:
    """Render an arena error as ``{"detail": message}``."""
    del request
    status = status_for(exc) if isinstance(exc, ArenaError) else 400
    return JSONResponse(
        status_code=status, content={"detail": str(exc), "type": type(exc).__name__}
    )


def install(app: FastAPI) -> None:
    """Register the error handler on ``app``."""
    app.add_exception_handler(ArenaError, _arena_error)
