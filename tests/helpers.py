"""Test doubles shared across the suite: a stepping clock and stub engines."""

from datetime import UTC, datetime, timedelta

import chess

from chess_arena.application.engines.positions import board_of
from chess_arena.application.ports.engines import EnginePort, MoveChoice, MoveRequest
from chess_arena.domain.players import Opponent, PlayerRef


class StepClock:
    """A clock that advances one second per call."""

    def __init__(self) -> None:
        self.current = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


class FirstMoveEngine:
    """Plays the first legal move in python-chess order."""

    def choose(self, request: MoveRequest) -> MoveChoice:
        board = board_of(request)
        return MoveChoice(uci=next(iter(board.legal_moves)).uci())

    def close(self) -> None:
        return None


class ScriptedEngine:
    """Plays a fixed list of moves, then raises."""

    def __init__(self, moves: list[str]) -> None:
        self._moves = list(moves)

    def choose(self, request: MoveRequest) -> MoveChoice:
        del request
        if not self._moves:
            msg = "script exhausted"
            raise ValueError(msg)
        return MoveChoice(uci=self._moves.pop(0))

    def close(self) -> None:
        return None


class StubFactory:
    """Engine factory that always seats ``FirstMoveEngine``."""

    def __init__(self) -> None:
        self.opened: list[PlayerRef] = []

    def open(self, player: PlayerRef, seed: int, max_seconds: float | None = None) -> EnginePort:
        del seed, max_seconds
        self.opened.append(player)
        return FirstMoveEngine()

    def opponents(self) -> tuple[Opponent, ...]:
        return ()


def fools_mate_board() -> chess.Board:
    """Return the position where Black mates in one with Qh4#."""
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        _ = board.push_uci(uci)
    return board
