"""Play moves with a trained value network inside the alpha-beta search."""

import math

import chess
import numpy as np

from chess_arena.application.engines.evaluation import evaluate, mop_up
from chess_arena.application.engines.positions import board_of
from chess_arena.application.engines.search import AlphaBeta, SearchSettings
from chess_arena.application.learning.features import feature_indices
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.ports.engines import MoveChoice, MoveRequest
from chess_arena.domain.recipes import SearchSpec

_VALUE_CLIP = 0.995


def value_to_cp(value: float, scale_cp: float) -> float:
    """Invert ``tanh(cp / scale)``: map a value in (-1, 1) to centipawns."""
    clipped = max(-_VALUE_CLIP, min(_VALUE_CLIP, value))
    return scale_cp * math.atanh(clipped)


class NetworkEvaluator:
    """Blend the network's value with the hand-written evaluation."""

    def __init__(self, network: ValueNetwork, material_blend: float, scale_cp: float) -> None:
        """Bind the network, the blend weight (1 = hand-written only), and the cp scale."""
        self._network = network
        self._blend = material_blend
        self._scale = scale_cp

    def evaluate(self, board: chess.Board) -> float:
        """Return centipawns for the side to move (mop-up technique always included)."""
        if self._blend >= 1.0:
            return evaluate(board)
        learned = value_to_cp(self._network.value(feature_indices(board)), self._scale)
        if self._blend <= 0.0:
            technique = mop_up(board)
            return learned + (technique if board.turn == chess.WHITE else -technique)
        return (1.0 - self._blend) * learned + self._blend * evaluate(board)


class ModelPlayer:
    """A trained model seated as a player."""

    def __init__(
        self,
        network: ValueNetwork,
        search: SearchSpec,
        scale_cp: float,
        rng: np.random.Generator,
        max_seconds: float | None = None,
    ) -> None:
        """Bind the network, its search settings, and an optional per-move cap."""
        seconds = search.max_seconds_per_move
        if max_seconds is not None:
            seconds = min(seconds, max_seconds)
        self._search = AlphaBeta(
            NetworkEvaluator(network, search.material_blend, scale_cp),
            SearchSettings(
                depth=search.depth,
                quiescence_depth=search.quiescence_depth,
                max_seconds=seconds,
            ),
            rng,
        )

    def choose(self, request: MoveRequest) -> MoveChoice:
        """Return the searched move and diagnostics."""
        result = self._search.search(board_of(request), request.seconds)
        return MoveChoice(
            uci=result.move, score_cp=result.score, depth=result.depth, nodes=result.nodes
        )

    def close(self) -> None:
        """Nothing to release."""
