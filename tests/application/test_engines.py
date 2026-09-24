import chess
import numpy as np

from chess_arena.application.engines.bots import ExploringEngine, GreedyBot, MinimaxBot, RandomBot
from chess_arena.application.engines.evaluation import (
    MaterialEvaluator,
    evaluate,
    material_balance,
    mop_up,
    white_score,
)
from chess_arena.application.engines.model_player import ModelPlayer, NetworkEvaluator, value_to_cp
from chess_arena.application.engines.search import MATE, AlphaBeta, SearchSettings
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.ports.engines import MoveRequest
from chess_arena.domain.recipes import NetworkSpec, SearchSpec
from tests.helpers import FirstMoveEngine, fools_mate_board


def test_start_position_is_balanced_and_mirrored_positions_agree() -> None:
    assert white_score(chess.Board()) == 0
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    mirrored = board.mirror()
    assert evaluate(board) == evaluate(mirrored)


def test_material_counts_and_side_to_move_perspective() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert material_balance(board) == 900
    assert evaluate(board) > 800
    board.turn = chess.BLACK
    assert evaluate(board) < -800


def test_mop_up_rewards_cornering_a_bare_king() -> None:
    cornered = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    central = chess.Board("8/8/8/3k4/8/8/8/K6R w - - 0 1")
    assert mop_up(cornered) > mop_up(central) > 0
    assert mop_up(chess.Board()) == 0


def test_search_finds_mate_in_one() -> None:
    search = AlphaBeta(MaterialEvaluator(), SearchSettings(depth=2, max_seconds=10))
    result = search.search(fools_mate_board())
    assert result.move == "d8h4"
    assert result.score >= MATE - 10


def test_search_wins_a_hanging_queen() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
    result = AlphaBeta(MaterialEvaluator(), SearchSettings(depth=2, max_seconds=10)).search(board)
    assert result.move == "d2d5"


def test_search_respects_a_tiny_time_budget() -> None:
    search = AlphaBeta(MaterialEvaluator(), SearchSettings(depth=8, max_seconds=0.05))
    result = search.search(chess.Board())
    assert chess.Move.from_uci(result.move) in chess.Board().legal_moves
    assert result.depth < 8


def test_bots_always_return_legal_moves() -> None:
    rng = np.random.default_rng(3)
    request = MoveRequest(moves=("e2e4", "e7e5"))
    board = chess.Board()
    _ = board.push_uci("e2e4")
    _ = board.push_uci("e7e5")
    for bot in (RandomBot(rng), GreedyBot(rng), MinimaxBot(1, rng, seconds=2)):
        assert chess.Move.from_uci(bot.choose(request).uci) in board.legal_moves


def test_greedy_bot_mates_in_one_and_takes_material() -> None:
    rng = np.random.default_rng(0)
    board = fools_mate_board()
    request = MoveRequest(moves=tuple(move.uci() for move in board.move_stack))
    assert GreedyBot(rng).choose(request).uci == "d8h4"
    grab = MoveRequest(start_fen="4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
    assert GreedyBot(rng).choose(grab).uci == "d2d5"


def test_exploring_engine_defers_when_epsilon_is_zero() -> None:
    engine = ExploringEngine(FirstMoveEngine(), 0.0, np.random.default_rng(0))
    expected = FirstMoveEngine().choose(MoveRequest())
    assert engine.choose(MoveRequest()) == expected


def test_value_to_cp_inverts_tanh_and_clips() -> None:
    assert value_to_cp(0.0, 400) == 0.0
    assert abs(value_to_cp(np.tanh(0.5), 400) - 200) < 1e-6
    assert value_to_cp(1.0, 400) < 1300


def test_blend_of_one_matches_the_hand_written_evaluation() -> None:
    network = ValueNetwork.initialize(NetworkSpec(hidden=(16,)), np.random.default_rng(0))
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert NetworkEvaluator(network, 1.0, 400).evaluate(board) == evaluate(board)


def test_model_player_plays_a_legal_move() -> None:
    network = ValueNetwork.initialize(NetworkSpec(hidden=(16,)), np.random.default_rng(0))
    player = ModelPlayer(
        network, SearchSpec(depth=1, max_seconds_per_move=0.5), 400.0, np.random.default_rng(0)
    )
    choice = player.choose(MoveRequest(moves=("d2d4",)))
    board = chess.Board()
    _ = board.push_uci("d2d4")
    assert chess.Move.from_uci(choice.uci) in board.legal_moves
