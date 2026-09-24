"""Human-versus-engine games: start, move, engine reply, resign, and replay."""

import secrets
from typing import Literal

import chess
import numpy as np
from pydantic import Field

from chess_arena.application.engines.positions import board_from, finished
from chess_arena.application.play.views import replay_of, view_of
from chess_arena.application.ports.engines import EngineFactoryPort, MoveRequest
from chess_arena.application.ports.runtime import ClockPort
from chess_arena.application.ports.stores import GameStorePort
from chess_arena.domain.contract import ArenaModel, UciMove
from chess_arena.domain.errors import GameOverError, IllegalMoveError
from chess_arena.domain.games import (
    Color,
    GameContext,
    GameKind,
    GameRecord,
    GameReplay,
    GameResult,
    GameView,
    Termination,
)
from chess_arena.domain.players import PlayerKind, PlayerRef

HUMAN = PlayerRef(kind=PlayerKind.HUMAN, player_id="human")


class NewGame(ArenaModel):
    """Start a game against an engine or model."""

    opponent: PlayerRef
    human_color: Literal["white", "black", "random"] = "white"


class HumanMove(ArenaModel):
    """A move submitted by the human player."""

    uci: UciMove


class EngineReply(ArenaModel):
    """Settings for the engine's reply move in human games."""

    seconds: float = Field(default=2.0, gt=0, le=60)


class GameService:
    """Play and replay human games."""

    def __init__(
        self,
        games: GameStorePort,
        engines: EngineFactoryPort,
        clock: ClockPort,
        reply: EngineReply | None = None,
    ) -> None:
        """Bind the game store, engine factory, clock, and reply limits."""
        self._games = games
        self._engines = engines
        self._clock = clock
        self._reply = reply or EngineReply()

    def start(self, request: NewGame) -> GameView:
        """Create a game; the engine moves first when it has White."""
        if request.opponent.kind is PlayerKind.HUMAN:
            msg = "choose an engine or a model as the opponent"
            raise IllegalMoveError(msg)
        color = request.human_color
        if color == "random":
            color = "white" if secrets.randbelow(2) == 0 else "black"
        white, black = (HUMAN, request.opponent) if color == "white" else (request.opponent, HUMAN)
        now = self._clock.now()
        record = GameRecord(
            game_id=f"hg-{now:%Y%m%dt%H%M%S}-{secrets.token_hex(3)}",
            white=white,
            black=black,
            context=GameContext(kind=GameKind.HUMAN),
            created_at=now,
        )
        if white != HUMAN:
            record = self._engine_move(record)
        self._games.save(record)
        return view_of(record)

    def view(self, game_id: str) -> GameView:
        """Return the current view of a game."""
        return view_of(self._games.get(game_id))

    def move(self, game_id: str, move: HumanMove) -> GameView:
        """Apply the human's move, then let the engine reply if the game goes on."""
        record = self._games.get(game_id)
        if record.finished:
            msg = "the game is already over"
            raise GameOverError(msg)
        board = board_from(record.start_fen, record.moves)
        human_turn = Color.WHITE if board.turn == chess.WHITE else Color.BLACK
        if record.player(human_turn) != HUMAN:
            msg = "it is not your move"
            raise IllegalMoveError(msg)
        parsed = chess.Move.from_uci(move.uci)
        if parsed not in board.legal_moves:
            msg = f"{move.uci} is not legal here"
            raise IllegalMoveError(msg)
        record = self._advance(record, move.uci)
        if not record.finished:
            record = self._engine_move(record)
        self._games.save(record)
        return view_of(record)

    def resign(self, game_id: str) -> GameView:
        """End the game as a loss for the human."""
        record = self._games.get(game_id)
        if record.finished:
            msg = "the game is already over"
            raise GameOverError(msg)
        human = Color.WHITE if record.white == HUMAN else Color.BLACK
        record = record.model_copy(
            update={
                "result": GameResult.win_for(human.opposite),
                "termination": Termination.RESIGNATION,
                "finished_at": self._clock.now(),
            }
        )
        self._games.save(record)
        return view_of(record)

    def replay(self, game_id: str) -> GameReplay:
        """Return every position of a game."""
        return replay_of(self._games.get(game_id))

    def _advance(self, record: GameRecord, uci: str) -> GameRecord:
        """Append ``uci`` and close the game if it ended."""
        moves = (*record.moves, uci)
        outcome = finished(board_from(record.start_fen, moves))
        if outcome is None:
            return record.model_copy(update={"moves": moves})
        result, termination = outcome
        return record.model_copy(
            update={
                "moves": moves,
                "result": result,
                "termination": termination,
                "finished_at": self._clock.now(),
            }
        )

    def _engine_move(self, record: GameRecord) -> GameRecord:
        """Ask the engine seated on the side to move for its reply."""
        board = board_from(record.start_fen, record.moves)
        side = Color.WHITE if board.turn == chess.WHITE else Color.BLACK
        seed = int(np.random.default_rng().integers(2**31))
        engine = self._engines.open(record.player(side), seed)
        try:
            choice = engine.choose(
                MoveRequest(
                    start_fen=record.start_fen, moves=record.moves, seconds=self._reply.seconds
                )
            )
        finally:
            engine.close()
        return self._advance(record, choice.uci)
