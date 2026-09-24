"""Stockfish over UCI: a (strength-limited) player and a position scorer.

Strength limiting uses Stockfish's own ``UCI_LimitStrength`` and ``UCI_Elo``
options (1320 to 3190). Every engine runs single-threaded with a small hash so
several games can run side by side on one machine.
"""

import shutil
from pathlib import Path

import chess
import chess.engine

from chess_arena.application.engines.positions import board_of
from chess_arena.application.ports.engines import MoveChoice, MoveRequest
from chess_arena.domain.errors import EngineUnavailableError

_MATE_CP = 32_000
_HASH_MB = 16
_FALLBACK_PATHS = (
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/games/stockfish",
)


def find_stockfish(configured: str | None) -> str | None:
    """Return a usable Stockfish path: the configured one, PATH, or a common location."""
    if configured:
        return configured if Path(configured).is_file() else shutil.which(configured)
    found = shutil.which("stockfish")
    if found:
        return found
    return next((path for path in _FALLBACK_PATHS if Path(path).is_file()), None)


def _launch(path: str) -> chess.engine.SimpleEngine:
    """Start Stockfish or raise ``EngineUnavailableError``."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
    except (OSError, chess.engine.EngineError) as exc:
        msg = f"could not start Stockfish at {path}: {exc}"
        raise EngineUnavailableError(msg) from exc
    engine.configure({"Threads": 1, "Hash": _HASH_MB})
    return engine


class StockfishEngine:
    """A Stockfish player at a fixed Elo (or full strength) and move time."""

    def __init__(self, path: str, elo: int | None, move_time_ms: int) -> None:
        """Start the process and apply the strength limit."""
        self._engine = _launch(path)
        self._move_time = move_time_ms / 1000.0
        if elo is not None:
            self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return Stockfish's move within the move-time limit."""
        seconds = self._move_time
        if request.seconds is not None:
            seconds = min(seconds, request.seconds)
        result = self._engine.play(board_of(request), chess.engine.Limit(time=seconds))
        if result.move is None:
            msg = "Stockfish returned no move"
            raise RuntimeError(msg)
        return MoveChoice(uci=result.move.uci())

    def close(self) -> None:
        """Stop the Stockfish process."""
        try:
            self._engine.quit()
        except (chess.engine.EngineError, TimeoutError, OSError):
            self._engine.close()


class StockfishOpener:
    """Open Stockfish players for data games."""

    def __init__(self, path: str) -> None:
        """Bind the binary path."""
        self._path = path

    def open(self, elo: int | None, move_time_ms: int) -> StockfishEngine:
        """Return a new Stockfish player."""
        return StockfishEngine(self._path, elo, move_time_ms)


class StockfishScorer:
    """Score positions with a fixed-depth Stockfish search."""

    def __init__(self, path: str) -> None:
        """Start one full-strength Stockfish process."""
        self._engine = _launch(path)

    def score(self, fen: str, depth: int) -> float:
        """Return centipawns for the side to move; mates map to +-32000."""
        board = chess.Board(fen)
        if board.is_game_over():
            return 0.0
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score is None:
            return 0.0
        return float(score.pov(board.turn).score(mate_score=_MATE_CP))

    def close(self) -> None:
        """Stop the Stockfish process."""
        try:
            self._engine.quit()
        except (chess.engine.EngineError, TimeoutError, OSError):
            self._engine.close()


def stockfish_version(path: str) -> str:
    """Return the engine's reported name, for example ``Stockfish 19``."""
    engine = _launch(path)
    try:
        return str(engine.id.get("name", "Stockfish"))
    finally:
        engine.quit()
