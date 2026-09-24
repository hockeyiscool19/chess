# Agent contract

Chess Arena: a web app and engine-ladder backend. The autoresearcher monorepo's
chess institute consumes it as the `apps/chess` git submodule, through the
`chess-arena` CLI and the HTTP API. Changes to the CLI envelope
(`chess-arena-cli-v1`), the recipe schema, or `/api/*` are therefore changes to
that institute. Keep them additive, or bump the version.

- **Pydantic, strongly typed.** Structured values are frozen Pydantic v2 models
  (`ArenaModel`). No `dataclass`, `TypedDict`, `NamedTuple`, untyped `dict`, or
  `Any` as a domain, port, or API shape. Settings use `pydantic-settings`.
- **Docstring every function** in `src/`.
- **OpenAPI compliant.** The HTTP API is FastAPI with OpenAPI 3 enabled (`/docs`).
- **Linted.** Ruff (lint and format) and Pyright strict. Do not weaken either config.
- **Files of 400 lines or fewer**, including JS and CSS. Split by seam before the cap.
- **Parallel unit tests** (`pytest -n auto`). Use fakes and `tmp_path`, never the
  network. Stockfish tests skip without the binary.
- **uv** for install, lock, and run.

## Architecture

Hexagonal. Domain has no I/O. Application depends on domain and ports only.
Adapters implement ports. `adapters/inbound/bootstrap.py` (`Arena`) is the only
composition root.

    src/chess_arena/domain/                  frozen models and pure rules (ladder score, Elo, recipes)
    src/chess_arena/application/ports/       engines, stores, runtime, training protocols
    src/chess_arena/application/engines/     alpha-beta search, bots, model player (pure CPU)
    src/chess_arena/application/learning/    features, numpy MLP, Adam, fit, npz blobs
    src/chess_arena/application/training/    data games, labels (TD(lambda), teacher), trainer
    src/chess_arena/application/ladder/      gauntlet with early decisions, openings
    src/chess_arena/application/play/        referee, human games, matches, views
    src/chess_arena/adapters/inbound/        cli/, api/ (FastAPI + static web app), settings, bootstrap
    src/chess_arena/adapters/outbound/       Stockfish, engine factory, file stores, process pool, ladders

Recipes are bounded data. An automated researcher may propose any valid recipe,
but it cannot run code in this process. Model uploads refuse pickled arrays.
Keep it that way.

## Gates

    uv run python scripts/gates.py

## Notes

- python-chess keeps a non-daemon thread per open engine. Always close Stockfish
  engines inside the task that opened them, or worker processes will never exit.
- Worker processes use the `spawn` start method. Pool entry points must stay
  top-level functions.
