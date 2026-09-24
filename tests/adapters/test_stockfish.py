"""Stockfish integration; skipped when no Stockfish binary is installed."""

import chess
import pytest

from chess_arena.adapters.outbound.engines.stockfish import (
    StockfishEngine,
    StockfishScorer,
    find_stockfish,
    stockfish_version,
)
from chess_arena.application.ports.engines import MoveRequest

STOCKFISH = find_stockfish(None)
pytestmark = pytest.mark.skipif(STOCKFISH is None, reason="Stockfish is not installed")


def _path() -> str:
    assert STOCKFISH is not None
    return STOCKFISH


def test_limited_stockfish_plays_legal_moves() -> None:
    engine = StockfishEngine(_path(), elo=1320, move_time_ms=20)
    try:
        choice = engine.choose(MoveRequest(moves=("e2e4",)))
    finally:
        engine.close()
    board = chess.Board()
    _ = board.push_uci("e2e4")
    assert chess.Move.from_uci(choice.uci) in board.legal_moves


def test_scorer_sees_a_queen_advantage() -> None:
    scorer = StockfishScorer(_path())
    try:
        assert scorer.score("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1", depth=6) > 500
        assert scorer.score("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1", depth=6) < -500
    finally:
        scorer.close()


def test_version_names_stockfish() -> None:
    assert "Stockfish" in stockfish_version(_path())
