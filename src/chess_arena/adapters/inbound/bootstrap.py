"""Composition root: wire stores, engines, the process pool, and the services."""

from pathlib import Path

from chess_arena.adapters.inbound.settings import ArenaSettings
from chess_arena.adapters.outbound.clock import SystemClock
from chess_arena.adapters.outbound.engines.factory import EngineFactory
from chess_arena.adapters.outbound.engines.stockfish import find_stockfish
from chess_arena.adapters.outbound.ladders import PackagedLadders
from chess_arena.adapters.outbound.parallel import InlinePool, ProcessPool, WorkerConfig
from chess_arena.adapters.outbound.store.files import (
    FileGameStore,
    FileJobStore,
    FileModelStore,
)
from chess_arena.application.ladder.service import LadderService, LadderStores
from chess_arena.application.play.games import EngineReply, GameService
from chess_arena.application.play.matches import MatchService
from chess_arena.application.ports.training import ModelSinkPort
from chess_arena.application.training.service import TrainingPorts, TrainingService


class Arena:
    """Every service the CLI and the HTTP API use, sharing one pool."""

    def __init__(self, settings: ArenaSettings, *, inline: bool = False) -> None:
        """Build stores, the engine factory, the worker pool, and the services."""
        home = Path(settings.home).resolve()
        home.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.home = home
        self.stockfish = find_stockfish(settings.stockfish)
        self.clock = SystemClock()
        self.ladders = PackagedLadders()
        self.games = FileGameStore(home)
        self.models = FileModelStore(home)
        self.jobs = FileJobStore(home)
        self.engines = EngineFactory(self.ladders, self.models, self.stockfish)
        config = WorkerConfig(home=str(home), stockfish=self.stockfish)
        self.pool = (
            InlinePool(config)
            if inline or settings.workers == 1
            else ProcessPool(config, settings.workers)
        )
        self.ladder = LadderService(
            LadderStores(self.ladders, self.models, self.jobs, self.games), self.pool, self.clock
        )
        self.matches = MatchService(self.jobs, self.games, self.pool, self.clock)
        self.play = GameService(
            self.games,
            self.engines,
            self.clock,
            EngineReply(seconds=settings.human_move_seconds),
        )

    def trainer(self, sink: ModelSinkPort) -> TrainingService:
        """Return a training service that writes into ``sink``."""
        return TrainingService(TrainingPorts(self.pool, self.pool, sink), self.clock)

    def close(self) -> None:
        """Stop worker processes."""
        self.pool.shutdown()
