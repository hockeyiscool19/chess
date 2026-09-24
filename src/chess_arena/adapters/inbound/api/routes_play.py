"""Opponents, human games, replays, and the games list."""

from typing import Annotated

from fastapi import APIRouter, Query

from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.application.play.games import HumanMove, NewGame
from chess_arena.domain.games import GameKind, GameRecord, GameReplay, GameView
from chess_arena.domain.ladder import LadderSpec
from chess_arena.domain.players import Opponent


def play_router(arena: Arena) -> APIRouter:
    """Return routes for opponents, ladders, and games."""
    router = APIRouter(prefix="/api", tags=["play"])

    @router.get("/opponents")
    def opponents() -> list[Opponent]:
        """List every engine rung and stored model a human can play."""
        return list(arena.engines.opponents())

    @router.get("/ladders")
    def ladders() -> list[LadderSpec]:
        """List the ladder protocols."""
        return list(arena.ladders.ladders())

    @router.get("/ladders/{ladder_id}")
    def ladder(ladder_id: str) -> LadderSpec:
        """Return one ladder protocol."""
        return arena.ladders.ladder(ladder_id)

    @router.post("/games", status_code=201)
    def start(request: NewGame) -> GameView:
        """Start a human game; the engine replies at once if it has White."""
        return arena.play.start(request)

    @router.get("/games")
    def games(
        kind: GameKind | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50
    ) -> list[GameRecord]:
        """List recent games, newest first."""
        return list(arena.games.recent(kind, limit))

    @router.get("/games/{game_id}")
    def game(game_id: str) -> GameView:
        """Return the current position of a game."""
        return arena.play.view(game_id)

    @router.post("/games/{game_id}/moves")
    def move(game_id: str, request: HumanMove) -> GameView:
        """Play the human's move; the engine's reply is included in the response."""
        return arena.play.move(game_id, request)

    @router.post("/games/{game_id}/resign")
    def resign(game_id: str) -> GameView:
        """Resign on behalf of the human."""
        return arena.play.resign(game_id)

    @router.get("/games/{game_id}/replay")
    def replay(game_id: str) -> GameReplay:
        """Return every position of a game and its PGN."""
        return arena.play.replay(game_id)

    return router
