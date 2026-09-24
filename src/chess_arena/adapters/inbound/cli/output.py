"""Versioned JSON envelopes and exit codes for ``chess-arena``.

Other programs (notably the chess institute in the autoresearcher monorepo) parse
these envelopes, so the ``schema`` string changes whenever their shape does.
"""

import json
import sys

from pydantic import BaseModel, JsonValue

from chess_arena.domain.contract import ArenaModel
from chess_arena.domain.errors import (
    ArenaError,
    ConflictError,
    EngineUnavailableError,
    NotFoundError,
)

SCHEMA = "chess-arena-cli-v1"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2


class CliError(ArenaModel):
    """Error type and message."""

    type: str
    message: str


class Envelope(ArenaModel):
    """One command's JSON result."""

    schema_version: str = SCHEMA
    ok: bool
    command: str
    data: JsonValue = None
    error: CliError | None = None


def emit(command: str, payload: BaseModel | JsonValue, *, as_json: bool, text: str) -> int:
    """Print a success envelope (JSON) or ``text`` and return exit code 0."""
    data: JsonValue = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    if as_json:
        envelope = Envelope(ok=True, command=command, data=data)
        print(envelope.model_dump_json())
    else:
        print(text)
    return EXIT_OK


def fail(command: str, exc: Exception, *, as_json: bool) -> int:
    """Print a failure envelope (JSON) or a message on stderr; return the exit code."""
    code = EXIT_FAILED if isinstance(exc, NotFoundError | ConflictError) else EXIT_INVALID
    if isinstance(exc, EngineUnavailableError):
        code = EXIT_INVALID
    if as_json:
        envelope = Envelope(
            ok=False, command=command, error=CliError(type=type(exc).__name__, message=str(exc))
        )
        print(envelope.model_dump_json())
    else:
        print(f"chess-arena {command}: {exc}", file=sys.stderr)
    return code


def expected(exc: Exception) -> bool:
    """Return whether ``exc`` is an expected, reportable failure."""
    return isinstance(exc, ArenaError | ValueError | OSError | json.JSONDecodeError)
