"""Game records, their outcomes, and the board view shown to players."""

from enum import StrEnum

from pydantic import AwareDatetime, Field

from chess_arena.domain.contract import STARTING_FEN, ArenaModel, Slug, UciMove
from chess_arena.domain.players import PlayerRef


class Color(StrEnum):
    """Side to move or side played."""

    WHITE = "white"
    BLACK = "black"

    @property
    def opposite(self) -> "Color":
        """Return the other color."""
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class GameResult(StrEnum):
    """PGN result token."""

    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    ONGOING = "*"

    def points_for(self, color: Color) -> float:
        """Return 1, 0.5, or 0 points for ``color``; an ongoing game scores 0."""
        if self is GameResult.DRAW:
            return 0.5
        if self is GameResult.ONGOING:
            return 0.0
        winner = Color.WHITE if self is GameResult.WHITE_WINS else Color.BLACK
        return 1.0 if winner is color else 0.0

    @classmethod
    def win_for(cls, color: Color) -> "GameResult":
        """Return the result in which ``color`` wins."""
        return cls.WHITE_WINS if color is Color.WHITE else cls.BLACK_WINS


class Termination(StrEnum):
    """Why a finished game ended."""

    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    SEVENTYFIVE_MOVES = "seventyfive_moves"
    FIVEFOLD_REPETITION = "fivefold_repetition"
    FIFTY_MOVES = "fifty_moves"
    THREEFOLD_REPETITION = "threefold_repetition"
    MAX_PLIES = "max_plies"
    RESIGNATION = "resignation"
    ENGINE_FAILURE = "engine_failure"


class GameKind(StrEnum):
    """Why a game was played; used to filter the games list."""

    HUMAN = "human"
    LADDER = "ladder"
    MATCH = "match"


class GameContext(ArenaModel):
    """Where a game came from. Ladder games name their run and rung."""

    kind: GameKind
    run_id: Slug | None = None
    rung_id: Slug | None = None
    opening: str | None = None


class GameRecord(ArenaModel):
    """A complete or in-progress game, stored move by move."""

    game_id: Slug
    white: PlayerRef
    black: PlayerRef
    start_fen: str = STARTING_FEN
    moves: tuple[UciMove, ...] = ()
    result: GameResult = GameResult.ONGOING
    termination: Termination | None = None
    context: GameContext
    created_at: AwareDatetime
    finished_at: AwareDatetime | None = None

    @property
    def finished(self) -> bool:
        """Return whether the game has a final result."""
        return self.result is not GameResult.ONGOING

    def player(self, color: Color) -> PlayerRef:
        """Return the participant playing ``color``."""
        return self.white if color is Color.WHITE else self.black


class GameView(ArenaModel):
    """Everything a board UI needs to render the current position."""

    game: GameRecord
    fen: str
    turn: Color
    legal_moves: tuple[UciMove, ...] = ()
    san_moves: tuple[str, ...] = ()
    last_move: UciMove | None = None
    check_square: str | None = None
    ply: int = Field(ge=0)


class ReplayFrame(ArenaModel):
    """One position in a game replay."""

    ply: int = Field(ge=0)
    fen: str
    san: str | None = None
    uci: UciMove | None = None


class GameReplay(ArenaModel):
    """Every position of a game so a viewer can step through it."""

    game: GameRecord
    frames: tuple[ReplayFrame, ...] = Field(min_length=1)
    pgn: str
