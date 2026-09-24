"""The referee: alternate two engines from an opening until the game ends.

An engine that raises, or returns an illegal move, forfeits the game with the
``engine_failure`` termination. Games longer than ``max_plies`` are drawn.
"""

from datetime import datetime

import chess
import chess.engine

from chess_arena.application.engines.positions import finished
from chess_arena.application.ports.engines import EnginePort, MoveRequest
from chess_arena.application.ports.runtime import GameAssignment
from chess_arena.domain.games import Color, GameRecord, GameResult, Termination


def _forfeit(board: chess.Board) -> tuple[GameResult, Termination]:
    """Return a loss for the side to move."""
    loser = Color.WHITE if board.turn == chess.WHITE else Color.BLACK
    return GameResult.win_for(loser.opposite), Termination.ENGINE_FAILURE


def play_game(
    assignment: GameAssignment,
    white: EnginePort,
    black: EnginePort,
    started_at: datetime,
    finished_at: datetime | None = None,
) -> GameRecord:
    """Play ``assignment`` to completion and return its record."""
    board = chess.Board()
    moves: list[str] = []
    for uci in assignment.opening.moves:
        _ = board.push_uci(uci)
        moves.append(uci)
    outcome: tuple[GameResult, Termination] | None = None
    while outcome is None:
        outcome = finished(board)
        if outcome is not None:
            break
        if len(moves) >= assignment.max_plies:
            outcome = (GameResult.DRAW, Termination.MAX_PLIES)
            break
        engine = white if board.turn == chess.WHITE else black
        try:
            choice = engine.choose(MoveRequest(moves=tuple(moves)))
            move = chess.Move.from_uci(choice.uci)
        except (ValueError, RuntimeError, OSError, chess.engine.EngineError):
            outcome = _forfeit(board)
            break
        if move not in board.legal_moves:
            outcome = _forfeit(board)
            break
        board.push(move)
        moves.append(choice.uci)
    result, termination = outcome
    return GameRecord(
        game_id=assignment.game_id,
        white=assignment.white,
        black=assignment.black,
        moves=tuple(moves),
        result=result,
        termination=termination,
        context=assignment.context,
        created_at=started_at,
        finished_at=finished_at or started_at,
    )
