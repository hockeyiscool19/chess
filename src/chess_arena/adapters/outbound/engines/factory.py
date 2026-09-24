"""Seat any player: built-in bots, Stockfish rungs, and stored models."""

from typing import TYPE_CHECKING

import numpy as np

from chess_arena.adapters.outbound.engines.stockfish import StockfishEngine
from chess_arena.application.engines.bots import GreedyBot, MinimaxBot, RandomBot
from chess_arena.application.engines.model_player import ModelPlayer
from chess_arena.application.ports.engines import EnginePort
from chess_arena.application.ports.stores import LadderCatalogPort, ModelStorePort
from chess_arena.domain.errors import EngineUnavailableError, NotFoundError
from chess_arena.domain.ladder import Rung
from chess_arena.domain.players import EngineFamily, EngineSpec, Opponent, PlayerKind, PlayerRef

if TYPE_CHECKING:
    from chess_arena.application.learning.network import ValueNetwork

_BOT_NOISE_CP = 10.0
_BOT_SECONDS = 5.0


class EngineFactory:
    """Open engines for ladder rungs and trained models."""

    def __init__(
        self, ladders: LadderCatalogPort, models: ModelStorePort, stockfish: str | None
    ) -> None:
        """Bind the ladder catalog, the model store, and the Stockfish path (if any)."""
        self._ladders = ladders
        self._models = models
        self._stockfish = stockfish
        self._networks: dict[str, ValueNetwork] = {}

    def open(self, player: PlayerRef, seed: int, max_seconds: float | None = None) -> EnginePort:
        """Return a fresh engine for ``player``."""
        rng = np.random.default_rng(seed)
        if player.kind is PlayerKind.ENGINE:
            return self._engine(self.rung(player.player_id).engine, rng)
        if player.kind is PlayerKind.MODEL:
            card = self._models.card(player.player_id)
            network = self._networks.get(card.model_id)
            if network is None:
                network = self._models.network(card.model_id)
                self._networks[card.model_id] = network
            return ModelPlayer(
                network, card.recipe.search, card.recipe.labels.eval_scale_cp, rng, max_seconds
            )
        msg = "a human cannot be opened as an engine"
        raise ValueError(msg)

    def rung(self, rung_id: str) -> Rung:
        """Return the rung with ``rung_id`` from any known ladder."""
        for ladder in self._ladders.ladders():
            for rung in ladder.rungs:
                if rung.rung_id == rung_id:
                    return rung
        msg = f"unknown engine {rung_id}"
        raise NotFoundError(msg)

    def opponents(self) -> tuple[Opponent, ...]:
        """Return every rung (with availability) followed by every stored model."""
        seen: set[str] = set()
        found: list[Opponent] = []
        for ladder in self._ladders.ladders():
            for rung in ladder.rungs:
                if rung.rung_id in seen:
                    continue
                seen.add(rung.rung_id)
                missing = rung.engine.family is EngineFamily.STOCKFISH and self._stockfish is None
                found.append(
                    Opponent(
                        player=PlayerRef(kind=PlayerKind.ENGINE, player_id=rung.rung_id),
                        label=rung.label,
                        description=rung.description,
                        nominal_elo=rung.nominal_elo,
                        available=not missing,
                        unavailable_reason="Stockfish is not installed" if missing else None,
                    )
                )
        found.extend(
            Opponent(
                player=PlayerRef(kind=PlayerKind.MODEL, player_id=card.model_id),
                label=card.label,
                description=card.notes[:200],
            )
            for card in self._models.cards()
        )
        return tuple(found)

    def _engine(self, spec: EngineSpec, rng: np.random.Generator) -> EnginePort:
        """Build the engine described by ``spec``."""
        match spec.family:
            case EngineFamily.RANDOM:
                return RandomBot(rng)
            case EngineFamily.GREEDY:
                return GreedyBot(rng)
            case EngineFamily.MINIMAX:
                depth = spec.depth or 1
                return MinimaxBot(depth, rng, noise_cp=_BOT_NOISE_CP, seconds=_BOT_SECONDS)
            case EngineFamily.STOCKFISH:
                if self._stockfish is None:
                    msg = "Stockfish is not installed; set CHESS_ARENA_STOCKFISH or install it"
                    raise EngineUnavailableError(msg)
                return StockfishEngine(self._stockfish, spec.elo, spec.move_time_ms)
