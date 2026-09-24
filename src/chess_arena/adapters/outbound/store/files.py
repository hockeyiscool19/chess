"""JSON and ``.npz`` files under one home directory, written atomically.

Layout::

    <home>/games/<game_id>.json
    <home>/models/<model_id>/card.json and weights.npz
    <home>/ladder-runs/<run_id>.json
    <home>/matches/<run_id>.json
"""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.learning.serialization import from_npz_bytes, to_npz_bytes
from chess_arena.domain.errors import ConflictError, NotFoundError
from chess_arena.domain.games import GameKind, GameRecord
from chess_arena.domain.jobs import JobStatus, LadderRun, MatchRun
from chess_arena.domain.ladder import LadderReport
from chess_arena.domain.models import ModelCard

CARD = "card.json"
WEIGHTS = "weights.npz"


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as stream:
            _ = stream.write(data)
        _ = Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_model(path: Path, model: BaseModel) -> None:
    """Write a Pydantic model as indented JSON."""
    write_atomic(path, model.model_dump_json(indent=2).encode())


def _newest_first(folder: Path) -> list[Path]:
    """Return JSON files in ``folder`` ordered by modification time, newest first."""
    if not folder.is_dir():
        return []
    files = [path for path in folder.glob("*.json") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime_ns, reverse=True)


class FileGameStore:
    """Games as one JSON file each."""

    def __init__(self, home: Path) -> None:
        """Bind the games folder."""
        self._folder = home / "games"

    def save(self, game: GameRecord) -> None:
        """Create or replace a game."""
        write_model(self._folder / f"{game.game_id}.json", game)

    def get(self, game_id: str) -> GameRecord:
        """Return one game."""
        path = self._folder / f"{game_id}.json"
        if not path.is_file():
            msg = f"unknown game {game_id}"
            raise NotFoundError(msg)
        return GameRecord.model_validate_json(path.read_bytes())

    def recent(self, kind: GameKind | None, limit: int) -> tuple[GameRecord, ...]:
        """Return the newest games, optionally of one kind."""
        found: list[GameRecord] = []
        for path in _newest_first(self._folder):
            game = GameRecord.model_validate_json(path.read_bytes())
            if kind is None or game.context.kind is kind:
                found.append(game)
            if len(found) >= limit:
                break
        return tuple(found)


class FileModelStore:
    """Model cards and weights, one folder per model."""

    def __init__(self, home: Path) -> None:
        """Bind the models folder."""
        self._folder = home / "models"

    def save(self, card: ModelCard, network: ValueNetwork) -> None:
        """Store a model; re-saving the same card is a no-op, a different one conflicts."""
        folder = self._folder / card.model_id
        if (folder / CARD).is_file():
            existing = ModelCard.model_validate_json((folder / CARD).read_bytes())
            if existing != card:
                msg = f"model {card.model_id} already exists with a different card"
                raise ConflictError(msg)
            return
        write_atomic(folder / WEIGHTS, to_npz_bytes(network))
        write_model(folder / CARD, card)

    def card(self, model_id: str) -> ModelCard:
        """Return one model card."""
        path = self._folder / model_id / CARD
        if not path.is_file():
            msg = f"unknown model {model_id}"
            raise NotFoundError(msg)
        return ModelCard.model_validate_json(path.read_bytes())

    def network(self, model_id: str) -> ValueNetwork:
        """Return one model's network."""
        card = self.card(model_id)
        arrays = from_npz_bytes((self._folder / model_id / WEIGHTS).read_bytes())
        return ValueNetwork.from_arrays(arrays, card.recipe.network.activation)

    def cards(self) -> tuple[ModelCard, ...]:
        """Return every model card, oldest first."""
        if not self._folder.is_dir():
            return ()
        cards = [
            ModelCard.model_validate_json((folder / CARD).read_bytes())
            for folder in self._folder.iterdir()
            if (folder / CARD).is_file()
        ]
        return tuple(sorted(cards, key=lambda card: (card.created_at, card.model_id)))


class DirectoryModelSink:
    """Write a trained model into one explicit folder (used by ``chess-arena train``)."""

    def __init__(self, folder: Path) -> None:
        """Bind the output folder."""
        self._folder = folder

    def save(self, card: ModelCard, network: ValueNetwork) -> None:
        """Write ``card.json`` and ``weights.npz``."""
        write_atomic(self._folder / WEIGHTS, to_npz_bytes(network))
        write_model(self._folder / CARD, card)


def read_model_folder(folder: Path) -> tuple[ModelCard, ValueNetwork]:
    """Read a model folder written by :class:`DirectoryModelSink`."""
    card_path = folder / CARD
    if not card_path.is_file():
        msg = f"{folder} has no {CARD}"
        raise NotFoundError(msg)
    card = ModelCard.model_validate_json(card_path.read_bytes())
    arrays = from_npz_bytes((folder / WEIGHTS).read_bytes())
    return card, ValueNetwork.from_arrays(arrays, card.recipe.network.activation)


class FileJobStore:
    """Ladder runs and matches as JSON files."""

    def __init__(self, home: Path) -> None:
        """Bind the job folders."""
        self._ladders = home / "ladder-runs"
        self._matches = home / "matches"

    def save_ladder_run(self, run: LadderRun) -> None:
        """Create or replace a ladder run."""
        write_model(self._ladders / f"{run.run_id}.json", run)

    def ladder_run(self, run_id: str) -> LadderRun:
        """Return one ladder run."""
        path = self._ladders / f"{run_id}.json"
        if not path.is_file():
            msg = f"unknown ladder run {run_id}"
            raise NotFoundError(msg)
        return LadderRun.model_validate_json(path.read_bytes())

    def ladder_runs(self, model_id: str | None) -> tuple[LadderRun, ...]:
        """Return ladder runs, newest first, optionally for one model."""
        runs = [
            LadderRun.model_validate_json(path.read_bytes())
            for path in _newest_first(self._ladders)
        ]
        return tuple(run for run in runs if model_id is None or run.request.model_id == model_id)

    def save_match(self, run: MatchRun) -> None:
        """Create or replace a match."""
        write_model(self._matches / f"{run.run_id}.json", run)

    def match(self, run_id: str) -> MatchRun:
        """Return one match."""
        path = self._matches / f"{run_id}.json"
        if not path.is_file():
            msg = f"unknown match {run_id}"
            raise NotFoundError(msg)
        return MatchRun.model_validate_json(path.read_bytes())

    def reports(self, model_id: str) -> tuple[LadderReport, ...]:
        """Return finished ladder reports for ``model_id``, newest first."""
        return tuple(
            run.report
            for run in self.ladder_runs(model_id)
            if run.status is JobStatus.COMPLETED and run.report is not None
        )
