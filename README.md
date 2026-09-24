# chess

Chess Arena: a web app where you can play a ladder of chess bots, from a
random mover up to full-strength Stockfish. It is also the backend a research
institute uses to train value-network models and measure them against that
ladder. Everything is Python (FastAPI, python-chess, numpy) with a build-free
web front end.

This repository is the `apps/chess` submodule of the
[autoresearcher](https://github.com/hockeyiscool19/autoresearcher) monorepo. Its
**chess institute** (`groups/chess_institute`) drives the `chess-arena` CLI and
this HTTP API. Its goal is a model that beats Stockfish.

## Quick start

```bash
brew install stockfish            # or: apt-get install stockfish (optional; enables rungs 6-14)
uv sync
uv run chess-arena serve          # http://127.0.0.1:8744
```

State (games, models, ladder runs) lives in `./var`, or in `CHESS_ARENA_HOME`.
The web app has four pages:

- **Play**: pick any bot, Stockfish level, or published model and play it on an
  accessible board (click, drag, or keyboard: arrow keys plus Enter).
- **Ladder**: the rungs and each model's best result against every rung.
- **Models**: the climb chart (ladder score by training order) and every model's
  recipe and lineage.
- **Games**: every human, ladder, and match game, with a step-by-step replay and PGN.

OpenAPI docs are at `/docs`.

## The ladder

`ladder-v1` has 14 rungs, played in order:

| # | Rung | How it plays | Nominal Elo |
| ---: | --- | --- | ---: |
| 1 | `random` | uniformly random legal moves | 250 |
| 2 | `greedy` | mates in one if it can, else grabs the most material | 500 |
| 3-5 | `minimax-1..3` | alpha-beta over material and piece-square terms | 800-1200 |
| 6-13 | `stockfish-1320..3190` | Stockfish with `UCI_LimitStrength`, 50 ms per move | 1320-3190 |
| 14 | `stockfish-full` | unrestricted Stockfish, 100 ms per move | ~3500 |

A model plays 8 games per rung, with colors alternating over book openings. It
**beats** a rung with at least 55% of the points. The gauntlet stops at the first
rung it cannot beat. Its **ladder score** is the number of rungs beaten plus its
point share on that last rung, so `6.4` means "beat six rungs and took 40% on
the seventh". Games run in parallel worker processes. A rung stops as soon as
its verdict is certain.

## Models and recipes

A model (`valuenet-v1`) is a numpy multilayer perceptron with 768 piece-square
inputs, seen from the side to move, and a tanh output: the expected result. It
plays inside the same alpha-beta search as the minimax bots. No deep-learning
framework is involved. As in `pong-rl`, the network, Adam, and the TD(lambda)
returns are written from scratch.

A model is produced by a **recipe**: bounded JSON, never code. A recipe covers:

- the network shape;
- which training games to play: random, noisy minimax, strength-limited
  Stockfish, and self-play by the network itself;
- how to label positions: game outcome, TD(lambda) returns, a heuristic
  teacher, or a Stockfish teacher, blended by `outcome_weight`;
- optimizer settings;
- how many play-label-fit rounds to run;
- the search it plays with (depth, quiescence, and `material_blend` between
  the network and the hand-written evaluation).

```bash
uv run chess-arena recipe schema                  # JSON Schema (the institute's agents fill this in)
uv run chess-arena recipe baseline --json         # hand-written evaluation, no training
uv run chess-arena train --recipe recipe.json --out models/m1 --model-id m1 --register
uv run chess-arena ladder run --model m1          # gauntlet from rung 1
uv run chess-arena models list
uv run chess-arena match --first model:m1 --second engine:stockfish-1320 --games 4
```

Every command takes `--json` and prints a versioned envelope
(`chess-arena-cli-v1`). Exit code 0 means success, 1 a failed operation, and
2 invalid input or a missing engine.

## HTTP API (used by the institute)

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | liveness, Stockfish path, model count |
| `GET /api/opponents` | every rung and model a human can face |
| `POST /api/models` | publish a model card plus base64 `.npz` weights (identical re-uploads are no-ops) |
| `GET /api/models`, `GET /api/models/{id}` | models with their best ladder results |
| `POST /api/ladder-runs` | queue a gauntlet, `202` plus a run to poll |
| `GET /api/ladder-runs/{id}` | progress and the (partial) report |
| `POST /api/matches`, `GET /api/matches/{id}` | automated matches |
| `POST /api/games`, `POST /api/games/{id}/moves`, `.../resign`, `.../replay` | human games |

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHESS_ARENA_HOME` | `./var` | games, models, and job records |
| `CHESS_ARENA_STOCKFISH` | auto-detected | Stockfish binary |
| `CHESS_ARENA_PORT` | `8744` | server port (`CHESS_ARENA_HOST` defaults to `127.0.0.1`) |
| `CHESS_ARENA_WORKERS` | CPU count minus 1 | game worker processes |
| `CHESS_ARENA_HUMAN_MOVE_SECONDS` | `2.0` | engine think time in human games |

## Checks

```bash
uv run python scripts/gates.py    # 400-line budget, Ruff lint and format, Pyright strict, pytest -n auto
```

The Stockfish tests skip when no binary is installed. Everything else uses
in-process fakes.

## Layout

```
src/chess_arena/domain/             ladder, recipes, games, players, jobs, models, Elo
src/chess_arena/application/        engines (search, bots, model player), learning (numpy MLP, Adam),
                                    training, ladder gauntlet, human games, matches, ports
src/chess_arena/adapters/inbound/   CLI (chess-arena), FastAPI app, static web app, composition root
src/chess_arena/adapters/outbound/  Stockfish over UCI, engine factory, file stores, process pool,
                                    bundled ladder specs
tests/                              mirrors the package
```

python-chess and Stockfish are GPL-3.0 projects. Stockfish is used as an
external program and is not bundled.
