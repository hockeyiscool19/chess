"""Ports for generating training games, labeling positions, and saving models."""

from enum import StrEnum
from typing import Protocol

from pydantic import Field

from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.ports.engines import EnginePort
from chess_arena.domain.contract import ArenaModel
from chess_arena.domain.models import ModelCard
from chess_arena.domain.recipes import Activation, TeacherKind


class DataGameKind(StrEnum):
    """Which players produce a batch of training games."""

    RANDOM = "random"
    BOTS = "bots"
    STOCKFISH = "stockfish"
    SELF_PLAY = "self_play"


class NetworkBlob(ArenaModel):
    """A network serialized as ``.npz`` bytes so it can cross process boundaries."""

    npz: bytes
    activation: Activation


class DataGameTask(ArenaModel):
    """Play ``games`` games of one kind and sample positions from each."""

    kind: DataGameKind
    games: int = Field(ge=1, le=10_000)
    seed: int = Field(ge=0)
    exploration: float = Field(ge=0.0, le=1.0)
    positions_per_game: int = Field(ge=1, le=400)
    keep_all_positions: bool = False
    max_plies: int = Field(default=200, ge=10, le=1000)
    network: NetworkBlob | None = None
    eval_scale_cp: float = Field(default=400.0, ge=50.0, le=2000.0)


class GamePositions(ArenaModel):
    """Positions from one training game with its final outcome.

    ``complete`` means every position of the game is present in order, which
    TD(lambda) labeling requires; otherwise the positions are a sample.
    """

    fens: tuple[str, ...]
    outcome_white: float = Field(ge=-1.0, le=1.0)
    complete: bool = False


class DataGeneratorPort(Protocol):
    """Play training games, possibly across worker processes."""

    def generate(self, tasks: tuple[DataGameTask, ...]) -> tuple[GamePositions, ...]:
        """Return the positions of every game in every task."""
        ...


class TeacherPort(Protocol):
    """Score positions in centipawns for the side to move."""

    def label(self, fens: tuple[str, ...], teacher: TeacherKind, depth: int) -> tuple[float, ...]:
        """Return one centipawn score per FEN."""
        ...


class StockfishOpenerPort(Protocol):
    """Open a (possibly strength-limited) Stockfish player."""

    def open(self, elo: int | None, move_time_ms: int) -> EnginePort:
        """Return a Stockfish engine; ``elo=None`` is full strength."""
        ...


class ModelSinkPort(Protocol):
    """Where a trained model is written."""

    def save(self, card: ModelCard, network: ValueNetwork) -> None:
        """Persist the card and weights."""
        ...
