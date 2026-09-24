"""Command handlers for ``chess-arena``; each returns a process exit code."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.adapters.inbound.cli.output import emit
from chess_arena.adapters.outbound.engines.stockfish import stockfish_version
from chess_arena.adapters.outbound.store.files import DirectoryModelSink, read_model_folder
from chess_arena.application.catalog import summarize
from chess_arena.application.training.service import TrainRequest
from chess_arena.domain.errors import ConflictError
from chess_arena.domain.jobs import LadderRunRequest, MatchRequest
from chess_arena.domain.players import PlayerRef
from chess_arena.domain.recipes import TrainingRecipe, baseline_recipe

if TYPE_CHECKING:
    from pydantic import JsonValue


def doctor(arena: Arena, args: argparse.Namespace) -> int:
    """Report the environment the arena will run in."""
    version = stockfish_version(arena.stockfish) if arena.stockfish else None
    data: JsonValue = {
        "home": str(arena.home),
        "stockfish": arena.stockfish,
        "stockfish_version": version,
        "workers": arena.settings.workers,
        "ladders": [spec.ladder_id for spec in arena.ladders.ladders()],
        "models": len(arena.models.cards()),
    }
    text = f"home {arena.home}\nstockfish {arena.stockfish or 'missing'} ({version or '-'})"
    return emit("doctor", data, as_json=args.json, text=text)


def engines(arena: Arena, args: argparse.Namespace) -> int:
    """List every opponent."""
    opponents = arena.engines.opponents()
    lines = [
        f"{item.player.key:32} {item.nominal_elo or '':>5} {'' if item.available else '(missing)'}"
        for item in opponents
    ]
    data: JsonValue = [item.model_dump(mode="json") for item in opponents]
    return emit("engines", data, as_json=args.json, text="\n".join(lines))


def ladder(arena: Arena, args: argparse.Namespace) -> int:
    """Show a ladder or run a model up it."""
    if args.action == "show":
        spec = arena.ladders.ladder(args.ladder)
        lines = [f"{rung.rung_id:18} {rung.nominal_elo:>5}  {rung.label}" for rung in spec.rungs]
        return emit("ladder show", spec, as_json=args.json, text="\n".join(lines))
    run = arena.ladder.run(
        LadderRunRequest(
            model_id=args.model,
            ladder_id=args.ladder,
            max_rungs=args.max_rungs,
            games_per_rung=args.games_per_rung,
            seed=args.seed,
        )
    )
    report = run.report
    text = "no report"
    if report is not None:
        rows = [
            f"{item.rung_id:18} +{item.wins} ={item.draws} -{item.losses}  "
            f"{'beaten' if item.passed else 'not beaten'}"
            for item in report.results
        ]
        rows.append(f"ladder score {report.ladder_score}  elo ~{report.elo_estimate}")
        text = "\n".join(rows)
    return emit("ladder run", run, as_json=args.json, text=text)


def models(arena: Arena, args: argparse.Namespace) -> int:
    """List, show, or import models."""
    if args.action == "list":
        summaries = [
            summarize(card, arena.jobs.reports(card.model_id)) for card in arena.models.cards()
        ]
        lines = [
            f"{item.card.model_id:40} score {item.best_ladder_score}  parent {item.card.parent_id}"
            for item in summaries
        ]
        data: JsonValue = [item.model_dump(mode="json") for item in summaries]
        return emit("models list", data, as_json=args.json, text="\n".join(lines))
    if args.action == "show":
        card = arena.models.card(args.model_id)
        reports = arena.jobs.reports(card.model_id)
        data = {
            "summary": summarize(card, reports).model_dump(mode="json"),
            "reports": [report.model_dump(mode="json") for report in reports],
        }
        return emit("models show", data, as_json=args.json, text=card.model_dump_json(indent=2))
    card, network = read_model_folder(Path(args.folder))
    arena.models.save(card, network)
    return emit("models import", card, as_json=args.json, text=f"imported {card.model_id}")


def recipe(args: argparse.Namespace) -> int:
    """Print the schema, validate a recipe, or print the baseline recipe."""
    if args.action == "schema":
        print(json.dumps(TrainingRecipe.model_json_schema(), indent=2))
        return 0
    if args.action == "baseline":
        base = baseline_recipe()
        return emit("recipe baseline", base, as_json=args.json, text=base.model_dump_json(indent=2))
    parsed = TrainingRecipe.model_validate_json(Path(args.file).read_bytes())
    data: JsonValue = {"digest": parsed.digest(), "recipe": parsed.model_dump(mode="json")}
    return emit("recipe validate", data, as_json=args.json, text=f"valid {parsed.digest()}")


def match(arena: Arena, args: argparse.Namespace) -> int:
    """Play a match and print the score from the first player's side."""
    run = arena.matches.run(
        MatchRequest(
            first=PlayerRef.parse(args.first),
            second=PlayerRef.parse(args.second),
            games=args.games,
            seed=args.seed,
        )
    )
    score = run.score
    text = f"{args.first} vs {args.second}: +{score.wins} ={score.draws} -{score.losses}"
    return emit("match", run, as_json=args.json, text=text)


def train(arena: Arena, args: argparse.Namespace) -> int:
    """Train a model folder from a recipe, optionally fine-tuning a parent folder."""
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        msg = f"{out} already exists and is not empty"
        raise ConflictError(msg)
    parsed = TrainingRecipe.model_validate_json(Path(args.recipe).read_bytes())
    parent_id = None
    parent_network = None
    if args.parent_dir:
        parent_card, parent_network = read_model_folder(Path(args.parent_dir))
        parent_id = parent_card.model_id
    card = arena.trainer(DirectoryModelSink(out)).train(
        TrainRequest(
            model_id=args.model_id,
            label=args.label or args.model_id,
            recipe=parsed,
            parent_id=parent_id,
            tags=tuple(args.tag),
            notes=args.notes,
            max_seconds=args.max_seconds,
        ),
        parent_network,
    )
    if args.register:
        stored_card, network = read_model_folder(out)
        arena.models.save(stored_card, network)
    summary = card.training
    text = f"trained {card.model_id} -> {out}"
    if summary is not None:
        text += f" ({summary.positions} positions, loss {summary.validation_loss})"
    return emit("train", card, as_json=args.json, text=text)
