"""Engine ports: choose a move for a position, and open engines by player reference."""

from typing import Protocol

from pydantic import Field

from chess_arena.domain.contract import STARTING_FEN, ArenaModel, UciMove
from chess_arena.domain.players import Opponent, PlayerRef


class MoveRequest(ArenaModel):
    """A position given as a start FEN plus the moves played since."""

    start_fen: str = STARTING_FEN
    moves: tuple[UciMove, ...] = ()
    seconds: float | None = Field(default=None, gt=0, le=600)


class MoveChoice(ArenaModel):
    """An engine's move with optional search diagnostics."""

    uci: UciMove
    score_cp: float | None = None
    depth: int | None = Field(default=None, ge=0)
    nodes: int | None = Field(default=None, ge=0)


class EnginePort(Protocol):
    """Something that plays moves: a bot, Stockfish, or a trained model."""

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return a legal move for the requested position."""
        ...

    def close(self) -> None:
        """Release processes or memory held by the engine."""
        ...


class EngineFactoryPort(Protocol):
    """Open engines for rung and model player references."""

    def open(self, player: PlayerRef, seed: int, max_seconds: float | None = None) -> EnginePort:
        """Return a fresh engine for ``player``; ``max_seconds`` caps a model's think time."""
        ...

    def opponents(self) -> tuple[Opponent, ...]:
        """Return every engine and model a human or a match may face."""
        ...
