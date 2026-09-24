"""Persistence ports for games, models, ladder runs, and matches."""

from typing import Protocol

from chess_arena.application.learning.network import ValueNetwork
from chess_arena.domain.games import GameKind, GameRecord
from chess_arena.domain.jobs import LadderRun, MatchRun
from chess_arena.domain.ladder import LadderReport, LadderSpec
from chess_arena.domain.models import ModelCard


class GameStorePort(Protocol):
    """Save and load game records."""

    def save(self, game: GameRecord) -> None:
        """Create or replace ``game``."""
        ...

    def get(self, game_id: str) -> GameRecord:
        """Return a game or raise ``NotFoundError``."""
        ...

    def recent(self, kind: GameKind | None, limit: int) -> tuple[GameRecord, ...]:
        """Return the newest games first, optionally of one kind."""
        ...


class ModelStorePort(Protocol):
    """Save and load model cards with their weights."""

    def save(self, card: ModelCard, network: ValueNetwork) -> None:
        """Store a model; an identical model is a no-op, a different one conflicts."""
        ...

    def card(self, model_id: str) -> ModelCard:
        """Return a model card or raise ``NotFoundError``."""
        ...

    def network(self, model_id: str) -> ValueNetwork:
        """Return a model's weights or raise ``NotFoundError``."""
        ...

    def cards(self) -> tuple[ModelCard, ...]:
        """Return every model card, oldest first."""
        ...


class JobStorePort(Protocol):
    """Save and load ladder runs, matches, and finished ladder reports."""

    def save_ladder_run(self, run: LadderRun) -> None:
        """Create or replace a ladder run."""
        ...

    def ladder_run(self, run_id: str) -> LadderRun:
        """Return a ladder run or raise ``NotFoundError``."""
        ...

    def ladder_runs(self, model_id: str | None) -> tuple[LadderRun, ...]:
        """Return ladder runs, newest first, optionally for one model."""
        ...

    def save_match(self, run: MatchRun) -> None:
        """Create or replace a match."""
        ...

    def match(self, run_id: str) -> MatchRun:
        """Return a match or raise ``NotFoundError``."""
        ...

    def reports(self, model_id: str) -> tuple[LadderReport, ...]:
        """Return finished ladder reports for one model, newest first."""
        ...


class LadderCatalogPort(Protocol):
    """Named ladder protocols."""

    def ladder(self, ladder_id: str) -> LadderSpec:
        """Return a ladder spec or raise ``NotFoundError``."""
        ...

    def ladders(self) -> tuple[LadderSpec, ...]:
        """Return every known ladder."""
        ...
