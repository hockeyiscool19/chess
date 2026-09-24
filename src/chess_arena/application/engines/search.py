"""Negamax alpha-beta search with quiescence, iterative deepening, and a clock.

The search is shared by the minimax bots (hand-written evaluation) and by the
trained models (network evaluation). Scores are centipawns for the side to move.
Root noise, when enabled, picks uniformly among moves within ``noise_cp`` of the
best so repeated games against the same opening do not all look identical.
"""

import time
from typing import Protocol

import chess
import numpy as np
from pydantic import Field

from chess_arena.application.engines.evaluation import PIECE_VALUES
from chess_arena.domain.contract import ArenaModel

MATE = 100_000.0
_INFINITY = 1_000_000.0
_MATE_BAND = 1_000.0
_CHECK_INTERVAL = 255
_DELTA_MARGIN = 200.0
_REPETITION_PLIES = 4
_FIFTY_MOVE_PLIES = 100


class Evaluator(Protocol):
    """Static evaluation in centipawns for the side to move."""

    def evaluate(self, board: chess.Board) -> float:
        """Return the score of ``board`` for ``board.turn``."""
        ...


class SearchSettings(ArenaModel):
    """Depth, quiescence depth, time budget, and root noise for one search."""

    depth: int = Field(default=2, ge=1, le=8)
    quiescence_depth: int = Field(default=4, ge=0, le=12)
    max_seconds: float = Field(default=1.0, gt=0, le=600)
    noise_cp: float = Field(default=0.0, ge=0, le=500)


class SearchResult(ArenaModel):
    """The chosen move, its score, the deepest completed iteration, and nodes."""

    move: str
    score: float
    depth: int = Field(ge=0)
    nodes: int = Field(ge=0)


class _SearchTimeoutError(Exception):
    """Raised inside the tree when the clock runs out."""


def _order_key(board: chess.Board, move: chess.Move) -> int:
    """Return a move-ordering key: promotions, then captures by MVV-LVA."""
    if move.promotion is not None:
        return 20_000 + PIECE_VALUES[move.promotion]
    victim = board.piece_type_at(move.to_square)
    if victim is None and board.is_en_passant(move):
        victim = chess.PAWN
    if victim is None:
        return 0
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    return 10_000 + 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker]


def ordered(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Return ``moves`` sorted so likely-best moves are searched first."""
    return sorted(moves, key=lambda move: _order_key(board, move), reverse=True)


class AlphaBeta:
    """One reusable searcher bound to an evaluator and settings."""

    def __init__(
        self,
        evaluator: Evaluator,
        settings: SearchSettings,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Bind the evaluator, limits, and the generator used for root noise."""
        self._evaluator = evaluator
        self._settings = settings
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._nodes = 0
        self._deadline = 0.0

    def search(self, board: chess.Board, max_seconds: float | None = None) -> SearchResult:
        """Return the best move found within the depth and time limits.

        ``max_seconds`` can only tighten the configured time budget.
        """
        moves = ordered(board, list(board.legal_moves))
        if not moves:
            msg = "cannot search a position without legal moves"
            raise ValueError(msg)
        budget = self._settings.max_seconds
        if max_seconds is not None:
            budget = min(budget, max_seconds)
        self._nodes = 0
        self._deadline = time.monotonic() + budget
        scored: list[tuple[chess.Move, float]] = [(moves[0], 0.0)]
        completed = 0
        for depth in range(1, self._settings.depth + 1):
            try:
                scored = self._root(board, moves, depth)
            except _SearchTimeoutError:
                break
            completed = depth
            moves = [move for move, _ in scored]
            if abs(scored[0][1]) >= MATE - _MATE_BAND:
                break
        move, score = self._pick(scored)
        return SearchResult(move=move.uci(), score=score, depth=completed, nodes=self._nodes)

    def _root(
        self, board: chess.Board, moves: list[chess.Move], depth: int
    ) -> list[tuple[chess.Move, float]]:
        """Score root moves with a window that keeps near-best moves exact."""
        if len(moves) == 1:
            return [(moves[0], 0.0)]
        results: list[tuple[chess.Move, float]] = []
        best = -_INFINITY
        noise = self._settings.noise_cp
        for move in moves:
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -_INFINITY, -(best - noise), 1)
            finally:
                _ = board.pop()
            results.append((move, score))
            best = max(best, score)
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def _pick(self, scored: list[tuple[chess.Move, float]]) -> tuple[chess.Move, float]:
        """Return the best move, or a uniform choice among those within the noise band."""
        best_score = scored[0][1]
        noise = self._settings.noise_cp
        if noise <= 0:
            return scored[0]
        near = [item for item in scored if item[1] >= best_score - noise]
        return near[int(self._rng.integers(len(near)))]

    def _tick(self) -> None:
        """Count a node and abort when the deadline has passed."""
        self._nodes += 1
        if not self._nodes & _CHECK_INTERVAL and time.monotonic() > self._deadline:
            raise _SearchTimeoutError

    def _negamax(
        self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int
    ) -> float:
        """Return the negamax score of ``board`` searched to ``depth`` plies."""
        self._tick()
        if _is_draw_by_rule(board):
            return 0.0
        if depth <= 0:
            return self._quiesce(board, alpha, beta, self._settings.quiescence_depth, ply)
        moves = list(board.legal_moves)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0.0
        best = -_INFINITY
        for move in ordered(board, moves):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            _ = board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _quiesce(
        self, board: chess.Board, alpha: float, beta: float, depth: int, ply: int
    ) -> float:
        """Resolve captures (or check evasions) so the evaluation is taken at rest."""
        self._tick()
        if board.is_check():
            return self._evasions(board, alpha, beta, depth, ply)
        stand = self._evaluator.evaluate(board)
        if stand >= beta or depth <= 0:
            return stand
        alpha = max(alpha, stand)
        for move in ordered(board, list(board.generate_legal_captures())):
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            if stand + PIECE_VALUES[victim] + _DELTA_MARGIN < alpha and move.promotion is None:
                continue
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, depth - 1, ply + 1)
            _ = board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _evasions(
        self, board: chess.Board, alpha: float, beta: float, depth: int, ply: int
    ) -> float:
        """Search every legal reply to a check; no replies means checkmate."""
        moves = list(board.legal_moves)
        if not moves:
            return -(MATE - ply)
        if depth <= 0:
            return self._evaluator.evaluate(board)
        best = -_INFINITY
        for move in ordered(board, moves):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, depth - 1, ply + 1)
            _ = board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best


def _is_draw_by_rule(board: chess.Board) -> bool:
    """Return whether a repetition or the fifty-move rule makes the node a draw."""
    clock = board.halfmove_clock
    if clock >= _FIFTY_MOVE_PLIES:
        return True
    return clock >= _REPETITION_PLIES and board.is_repetition(2)
