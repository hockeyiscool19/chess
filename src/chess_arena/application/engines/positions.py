"""Rebuild boards from move requests and describe finished positions."""

import chess

from chess_arena.application.ports.engines import MoveRequest
from chess_arena.domain.errors import IllegalMoveError
from chess_arena.domain.games import GameResult, Termination


def board_of(request: MoveRequest) -> chess.Board:
    """Return the board after replaying ``request.moves`` from its start FEN."""
    board = chess.Board(request.start_fen)
    for uci in request.moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            msg = f"illegal move {uci} in replayed game"
            raise IllegalMoveError(msg)
        board.push(move)
    return board


def board_from(start_fen: str, moves: tuple[str, ...]) -> chess.Board:
    """Return the board after ``moves`` from ``start_fen``."""
    return board_of(MoveRequest(start_fen=start_fen, moves=moves))


_TERMINATIONS: dict[chess.Termination, Termination] = {
    chess.Termination.CHECKMATE: Termination.CHECKMATE,
    chess.Termination.STALEMATE: Termination.STALEMATE,
    chess.Termination.INSUFFICIENT_MATERIAL: Termination.INSUFFICIENT_MATERIAL,
    chess.Termination.SEVENTYFIVE_MOVES: Termination.SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: Termination.FIVEFOLD_REPETITION,
    chess.Termination.FIFTY_MOVES: Termination.FIFTY_MOVES,
    chess.Termination.THREEFOLD_REPETITION: Termination.THREEFOLD_REPETITION,
}


def finished(board: chess.Board) -> tuple[GameResult, Termination] | None:
    """Return the result and reason when the game is over (draws claimed automatically)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    termination = _TERMINATIONS.get(outcome.termination, Termination.STALEMATE)
    if outcome.winner is None:
        return GameResult.DRAW, termination
    result = GameResult.WHITE_WINS if outcome.winner == chess.WHITE else GameResult.BLACK_WINS
    return result, termination
