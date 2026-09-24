from pathlib import Path

import numpy as np
import pytest

from chess_arena.adapters.outbound.store.files import FileGameStore
from chess_arena.application.engines.bots import RandomBot
from chess_arena.application.play.games import GameService, HumanMove, NewGame
from chess_arena.application.play.referee import play_game
from chess_arena.application.ports.runtime import GameAssignment, Opening
from chess_arena.domain.errors import GameOverError, IllegalMoveError
from chess_arena.domain.games import GameContext, GameKind, GameResult, Termination
from chess_arena.domain.players import PlayerKind, PlayerRef
from tests.helpers import FirstMoveEngine, ScriptedEngine, StepClock, StubFactory

ENGINE = PlayerRef(kind=PlayerKind.ENGINE, player_id="random")
OTHER = PlayerRef(kind=PlayerKind.ENGINE, player_id="greedy")


def _assignment(max_plies: int = 300, opening: Opening | None = None) -> GameAssignment:
    return GameAssignment(
        game_id="g-1",
        white=ENGINE,
        black=OTHER,
        max_plies=max_plies,
        opening=opening or Opening(name="Start position"),
        context=GameContext(kind=GameKind.MATCH),
    )


def test_referee_records_checkmate() -> None:
    clock = StepClock()
    white = ScriptedEngine(["f2f3", "g2g4"])
    black = ScriptedEngine(["e7e5", "d8h4"])
    record = play_game(_assignment(), white, black, clock.now())
    assert record.result is GameResult.BLACK_WINS
    assert record.termination is Termination.CHECKMATE
    assert record.moves == ("f2f3", "e7e5", "g2g4", "d8h4")


def test_referee_forfeits_an_illegal_or_failing_engine() -> None:
    clock = StepClock()
    illegal = play_game(_assignment(), ScriptedEngine(["e2e5"]), FirstMoveEngine(), clock.now())
    assert illegal.result is GameResult.BLACK_WINS
    assert illegal.termination is Termination.ENGINE_FAILURE
    crashed = play_game(_assignment(), FirstMoveEngine(), ScriptedEngine([]), clock.now())
    assert crashed.result is GameResult.WHITE_WINS


def test_referee_draws_at_the_ply_limit_and_plays_the_opening() -> None:
    opening = Opening(name="Italian", moves=("e2e4", "e7e5"))
    record = play_game(
        _assignment(max_plies=20, opening=opening),
        RandomBot(np.random.default_rng(1)),
        RandomBot(np.random.default_rng(2)),
        StepClock().now(),
    )
    assert record.moves[:2] == ("e2e4", "e7e5")
    assert record.result is GameResult.DRAW
    assert record.termination is Termination.MAX_PLIES


def test_human_game_flow(tmp_path: Path) -> None:
    factory = StubFactory()
    service = GameService(FileGameStore(tmp_path), factory, StepClock())
    view = service.start(NewGame(opponent=ENGINE, human_color="white"))
    assert view.turn.value == "white"
    assert "e2e4" in view.legal_moves
    after = service.move(view.game.game_id, HumanMove(uci="e2e4"))
    assert after.ply == 2
    assert after.turn.value == "white"
    with pytest.raises(IllegalMoveError):
        _ = service.move(view.game.game_id, HumanMove(uci="e2e4"))
    resigned = service.resign(view.game.game_id)
    assert resigned.game.result is GameResult.BLACK_WINS
    assert resigned.game.termination is Termination.RESIGNATION
    with pytest.raises(GameOverError):
        _ = service.move(view.game.game_id, HumanMove(uci="d2d4"))
    replay = service.replay(view.game.game_id)
    assert len(replay.frames) == 3
    assert '[Result "0-1"]' in replay.pgn


def test_engine_moves_first_when_the_human_takes_black(tmp_path: Path) -> None:
    factory = StubFactory()
    service = GameService(FileGameStore(tmp_path), factory, StepClock())
    view = service.start(NewGame(opponent=ENGINE, human_color="black"))
    assert view.ply == 1
    assert view.turn.value == "black"
    assert factory.opened == [ENGINE]


def test_humans_cannot_be_opponents(tmp_path: Path) -> None:
    service = GameService(FileGameStore(tmp_path), StubFactory(), StepClock())
    with pytest.raises(IllegalMoveError):
        _ = service.start(NewGame(opponent=PlayerRef(kind=PlayerKind.HUMAN, player_id="human")))
