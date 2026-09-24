"""Argument parser for ``chess-arena``."""

import argparse
from typing import Protocol


class _Commands(Protocol):
    """The ``add_parser`` half of argparse's sub-command action."""

    def add_parser(self, name: str, *, help: str) -> argparse.ArgumentParser:  # noqa: A002
        """Add one sub-command parser."""
        ...


def _json(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--json`` flag."""
    _ = parser.add_argument("--json", action="store_true", help="print a JSON envelope")


def _ladder(commands: _Commands) -> None:
    """Add ``ladder show`` and ``ladder run``."""
    ladder = commands.add_parser("ladder", help="show a ladder or run a model up it")
    actions = ladder.add_subparsers(dest="action", required=True)
    show = actions.add_parser("show", help="list the rungs of a ladder")
    _ = show.add_argument("--ladder", default="ladder-v1")
    _json(show)
    run = actions.add_parser("run", help="evaluate a stored model on the ladder")
    _ = run.add_argument("--model", required=True, help="model ID in the arena store")
    _ = run.add_argument("--ladder", default="ladder-v1")
    _ = run.add_argument("--max-rungs", type=int)
    _ = run.add_argument("--games-per-rung", type=int)
    _ = run.add_argument("--seed", type=int, default=0)
    _json(run)


def _models(commands: _Commands) -> None:
    """Add ``models list``, ``models show``, and ``models import``."""
    models = commands.add_parser("models", help="list, show, or import models")
    actions = models.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", help="list stored models with their best ladder score")
    _json(listing)
    show = actions.add_parser("show", help="show one model card and its ladder reports")
    _ = show.add_argument("model_id")
    _json(show)
    imported = actions.add_parser("import", help="copy a trained model folder into the store")
    _ = imported.add_argument("folder")
    _json(imported)


def _recipe(commands: _Commands) -> None:
    """Add ``recipe schema``, ``recipe validate``, and ``recipe baseline``."""
    recipe = commands.add_parser("recipe", help="training recipe schema and validation")
    actions = recipe.add_subparsers(dest="action", required=True)
    _ = actions.add_parser("schema", help="print the recipe JSON Schema")
    validate = actions.add_parser("validate", help="validate a recipe file")
    _ = validate.add_argument("file")
    _json(validate)
    baseline = actions.add_parser("baseline", help="print the baseline recipe")
    _json(baseline)


def build_parser() -> argparse.ArgumentParser:
    """Return the complete argument parser."""
    parser = argparse.ArgumentParser(
        prog="chess-arena",
        description="Chess web app and engine arena: play, ladder, match, and train.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the web app and HTTP API")
    _ = serve.add_argument("--host")
    _ = serve.add_argument("--port", type=int)
    _ = serve.add_argument("--home", help="state folder (defaults to CHESS_ARENA_HOME or ./var)")
    _ = serve.add_argument("--workers", type=int, help="game worker processes")

    doctor = commands.add_parser("doctor", help="report Stockfish, home, and worker settings")
    _json(doctor)

    engines = commands.add_parser("engines", help="list every opponent and its availability")
    _json(engines)

    _ladder(commands)
    _models(commands)
    _recipe(commands)

    match = commands.add_parser("match", help="play a match between two automated players")
    _ = match.add_argument("--first", required=True, help="engine:<rung> or model:<id>")
    _ = match.add_argument("--second", required=True, help="engine:<rung> or model:<id>")
    _ = match.add_argument("--games", type=int, default=2)
    _ = match.add_argument("--seed", type=int, default=0)
    _json(match)

    train = commands.add_parser("train", help="train a model folder from a recipe")
    _ = train.add_argument("--recipe", required=True, help="recipe JSON file")
    _ = train.add_argument("--out", required=True, help="new model folder")
    _ = train.add_argument("--model-id", required=True)
    _ = train.add_argument("--label", help="display name (defaults to the model ID)")
    _ = train.add_argument("--parent-dir", help="parent model folder to fine-tune")
    _ = train.add_argument("--tag", action="append", default=[], help="repeatable card tag")
    _ = train.add_argument("--notes", default="", help="free text stored on the card")
    _ = train.add_argument("--max-seconds", type=float, default=3600.0)
    _ = train.add_argument("--register", action="store_true", help="also import into the store")
    _json(train)
    return parser
