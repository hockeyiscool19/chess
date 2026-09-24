"""Clock and parallel game execution ports."""

from collections.abc import Generator
from datetime import datetime
from typing import Protocol

from pydantic import Field

from chess_arena.domain.contract import ArenaModel, Slug, UciMove
from chess_arena.domain.games import GameContext, GameRecord
from chess_arena.domain.players import PlayerRef


class ClockPort(Protocol):
    """Current time, injectable for tests."""

    def now(self) -> datetime:
        """Return an aware UTC timestamp."""
        ...


class Opening(ArenaModel):
    """A named sequence of opening moves played before engines take over."""

    name: str = Field(min_length=1, max_length=80)
    moves: tuple[UciMove, ...] = ()


class GameAssignment(ArenaModel):
    """One automated game to play."""

    game_id: Slug
    white: PlayerRef
    black: PlayerRef
    opening: Opening = Opening(name="Start position")
    max_plies: int = Field(default=300, ge=20, le=2000)
    seed: int = Field(default=0, ge=0)
    model_max_seconds: float | None = Field(default=None, gt=0, le=60)
    context: GameContext


class GameRunnerPort(Protocol):
    """Play automated games, possibly in parallel worker processes."""

    def play(self, assignments: tuple[GameAssignment, ...]) -> Generator[GameRecord]:
        """Yield finished games in completion order; closing the generator cancels the rest."""
        ...
