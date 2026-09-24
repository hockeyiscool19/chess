"""Who plays a game: a human, a ladder engine, or a trained model."""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from chess_arena.domain.contract import ArenaModel, ShortText, Slug


class PlayerKind(StrEnum):
    """The three kinds of participant the arena can seat."""

    HUMAN = "human"
    ENGINE = "engine"
    MODEL = "model"


class PlayerRef(ArenaModel):
    """A reference to one participant: a rung ID, a model ID, or ``human``."""

    kind: PlayerKind
    player_id: Slug

    @property
    def key(self) -> str:
        """Return ``kind:player_id``, the form used in URLs and CLI flags."""
        return f"{self.kind.value}:{self.player_id}"

    @classmethod
    def parse(cls, text: str) -> "PlayerRef":
        """Parse ``kind:player_id`` (for example ``engine:stockfish-1320``)."""
        kind, _, player_id = text.partition(":")
        return cls(kind=PlayerKind(kind), player_id=player_id)


class EngineFamily(StrEnum):
    """Built-in bots and the external UCI engine family."""

    RANDOM = "random"
    GREEDY = "greedy"
    MINIMAX = "minimax"
    STOCKFISH = "stockfish"


class EngineSpec(ArenaModel):
    """How to construct one opponent engine.

    ``depth`` applies to the minimax bot. ``elo`` limits Stockfish strength through
    ``UCI_LimitStrength``; ``None`` means full strength. ``move_time_ms`` caps
    Stockfish thinking time per move.
    """

    family: EngineFamily
    depth: int | None = Field(default=None, ge=1, le=6)
    elo: int | None = Field(default=None, ge=1320, le=3190)
    move_time_ms: int = Field(default=50, ge=1, le=10_000)

    @model_validator(mode="after")
    def parameters_match_family(self) -> Self:
        """Require depth only for minimax and Elo only for Stockfish."""
        if self.family is EngineFamily.MINIMAX and self.depth is None:
            msg = "minimax engines need a depth"
            raise ValueError(msg)
        if self.family is not EngineFamily.MINIMAX and self.depth is not None:
            msg = "only minimax engines take a depth"
            raise ValueError(msg)
        if self.family is not EngineFamily.STOCKFISH and self.elo is not None:
            msg = "only Stockfish takes an Elo limit"
            raise ValueError(msg)
        return self


class Opponent(ArenaModel):
    """A playable opponent as listed to humans and the institute."""

    player: PlayerRef
    label: ShortText
    description: str = ""
    nominal_elo: int | None = Field(default=None, ge=0, le=4000)
    available: bool = True
    unavailable_reason: str | None = None
