"""Built-in opponents for the lower ladder rungs and for training data.

``RandomBot`` plays any legal move. ``GreedyBot`` grabs the most valuable capture
available and otherwise plays randomly. ``MinimaxBot`` searches with the
hand-written evaluation. ``ExploringEngine`` wraps any engine and sometimes plays
a random move instead, which diversifies training games.
"""

import chess
import numpy as np

from chess_arena.application.engines.evaluation import PIECE_VALUES, MaterialEvaluator
from chess_arena.application.engines.positions import board_of
from chess_arena.application.engines.search import AlphaBeta, SearchSettings
from chess_arena.application.ports.engines import EnginePort, MoveChoice, MoveRequest


def _legal(board: chess.Board) -> list[chess.Move]:
    """Return legal moves or raise for a finished position."""
    moves = list(board.legal_moves)
    if not moves:
        msg = "no legal moves in this position"
        raise ValueError(msg)
    return moves


class RandomBot:
    """Uniformly random legal moves."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Bind the generator that makes games reproducible."""
        self._rng = rng

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return a random legal move."""
        moves = _legal(board_of(request))
        return MoveChoice(uci=moves[int(self._rng.integers(len(moves)))].uci())

    def close(self) -> None:
        """Nothing to release."""


def _gain(board: chess.Board, move: chess.Move) -> int:
    """Return the material captured or promoted by ``move``."""
    gain = 0
    if move.promotion is not None:
        gain += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    if victim is not None:
        gain += PIECE_VALUES[victim]
    elif board.is_en_passant(move):
        gain += PIECE_VALUES[chess.PAWN]
    return gain


class GreedyBot:
    """Take the biggest immediate material gain; mate in one if it sees it."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Bind the generator used to break ties."""
        self._rng = rng

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return a mating move, else a best-gain move chosen at random among ties."""
        board = board_of(request)
        moves = _legal(board)
        for move in moves:
            board.push(move)
            mate = board.is_checkmate()
            _ = board.pop()
            if mate:
                return MoveChoice(uci=move.uci())
        gains = [_gain(board, move) for move in moves]
        best = max(gains)
        choices = [move for move, gain in zip(moves, gains, strict=True) if gain == best]
        return MoveChoice(uci=choices[int(self._rng.integers(len(choices)))].uci())

    def close(self) -> None:
        """Nothing to release."""


class MinimaxBot:
    """Alpha-beta search at a fixed depth over the hand-written evaluation."""

    def __init__(
        self, depth: int, rng: np.random.Generator, noise_cp: float = 10.0, seconds: float = 5.0
    ) -> None:
        """Bind depth, root noise, and a per-move time cap."""
        self._search = AlphaBeta(
            MaterialEvaluator(),
            SearchSettings(depth=depth, quiescence_depth=4, max_seconds=seconds, noise_cp=noise_cp),
            rng,
        )

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return the searched move with its score, depth, and node count."""
        result = self._search.search(board_of(request), request.seconds)
        return MoveChoice(
            uci=result.move, score_cp=result.score, depth=result.depth, nodes=result.nodes
        )

    def close(self) -> None:
        """Nothing to release."""


class ExploringEngine:
    """Play a random legal move with probability ``epsilon``, else defer to ``inner``."""

    def __init__(self, inner: EnginePort, epsilon: float, rng: np.random.Generator) -> None:
        """Bind the wrapped engine and the exploration rate."""
        self._inner = inner
        self._epsilon = epsilon
        self._rng = rng

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return an exploratory or an engine move."""
        if self._epsilon > 0 and float(self._rng.random()) < self._epsilon:
            moves = _legal(board_of(request))
            return MoveChoice(uci=moves[int(self._rng.integers(len(moves)))].uci())
        return self._inner.choose(request)

    def close(self) -> None:
        """Close the wrapped engine."""
        self._inner.close()
