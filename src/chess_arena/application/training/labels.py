"""Turn game outcomes, TD(lambda) returns, and teacher scores into targets in [-1, 1].

All targets are from the side to move's point of view. For self-play games with
``td_lambda`` set, each position's outcome is replaced by the lambda-return of
the network's own evaluations along the game (TD-Gammon style): lambda = 1 is the
plain game result, lambda = 0 bootstraps entirely from the next position.
"""

import math

import chess
import numpy as np
from numpy.typing import NDArray

from chess_arena.application.engines.evaluation import MaterialEvaluator
from chess_arena.application.engines.search import AlphaBeta, SearchSettings
from chess_arena.application.learning.features import feature_indices
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.ports.training import GamePositions
from chess_arena.domain.recipes import LabelSpec

_WHITE_TO_MOVE = " w "
_HEURISTIC_SECONDS = 2.0


def white_to_move(fen: str) -> bool:
    """Return whether the FEN has White to move."""
    return _WHITE_TO_MOVE in fen


def mover_outcome(fen: str, outcome_white: float) -> float:
    """Return the game outcome from the perspective of the side to move."""
    return outcome_white if white_to_move(fen) else -outcome_white


def td_returns(game: GamePositions, network: ValueNetwork, td_lambda: float) -> list[float]:
    """Return per-position lambda-returns from the side to move's point of view."""
    white_values: list[float] = []
    for fen in game.fens:
        value = network.value(feature_indices(chess.Board(fen)))
        white_values.append(value if white_to_move(fen) else -value)
    returns = [0.0] * len(game.fens)
    following = game.outcome_white
    for index in range(len(game.fens) - 1, -1, -1):
        bootstrap = white_values[index + 1] if index + 1 < len(game.fens) else game.outcome_white
        following = (1.0 - td_lambda) * bootstrap + td_lambda * following
        returns[index] = following
    return [
        value if white_to_move(fen) else -value
        for fen, value in zip(game.fens, returns, strict=True)
    ]


def sample_indices(count: int, keep: int, rng: np.random.Generator) -> list[int]:
    """Return up to ``keep`` sorted positions among ``count`` (skipping the first two)."""
    eligible = list(range(min(2, count), count))
    if len(eligible) <= keep:
        return eligible
    chosen = rng.choice(len(eligible), size=keep, replace=False)
    return sorted(eligible[int(index)] for index in chosen)


def blend_targets(
    outcomes: list[float], teacher_cp: list[float] | None, spec: LabelSpec
) -> NDArray[np.float32]:
    """Return ``w * outcome + (1 - w) * tanh(cp / scale)`` for every position."""
    if teacher_cp is None:
        return np.asarray(outcomes, dtype=np.float32)
    weight = spec.outcome_weight
    values = [
        weight * outcome + (1.0 - weight) * math.tanh(cp / spec.eval_scale_cp)
        for outcome, cp in zip(outcomes, teacher_cp, strict=True)
    ]
    return np.asarray(values, dtype=np.float32)


class HeuristicTeacher:
    """Label positions with the hand-written evaluation after resolving captures."""

    def __init__(self) -> None:
        """Build a one-ply searcher over the material evaluation."""
        self._search = AlphaBeta(
            MaterialEvaluator(),
            SearchSettings(depth=1, quiescence_depth=6, max_seconds=_HEURISTIC_SECONDS),
        )

    def score(self, fen: str) -> float:
        """Return centipawns for the side to move (0 for finished positions)."""
        board = chess.Board(fen)
        if board.is_game_over():
            return 0.0
        return self._search.search(board).score
