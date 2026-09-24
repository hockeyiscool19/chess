"""Run games, training-data tasks, and teacher labeling in worker processes.

Each worker builds its own engine factory from a small picklable config, so
nothing heavy crosses process boundaries except game assignments, positions,
and results. The ``Inline*`` classes do the same
work in-process for tests and single-core runs.
"""

import multiprocessing
from collections.abc import Generator, Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from chess_arena.adapters.outbound.engines.factory import EngineFactory
from chess_arena.adapters.outbound.engines.stockfish import StockfishOpener, StockfishScorer
from chess_arena.adapters.outbound.ladders import PackagedLadders
from chess_arena.adapters.outbound.store.files import FileModelStore
from chess_arena.application.play.referee import play_game
from chess_arena.application.ports.runtime import GameAssignment
from chess_arena.application.ports.training import DataGameTask, GamePositions
from chess_arena.application.training.data import play_data_games
from chess_arena.application.training.labels import HeuristicTeacher
from chess_arena.domain.contract import ArenaModel
from chess_arena.domain.errors import EngineUnavailableError
from chess_arena.domain.games import GameRecord
from chess_arena.domain.players import PlayerKind
from chess_arena.domain.recipes import TeacherKind

_LABEL_CHUNK = 400


class WorkerConfig(ArenaModel):
    """What a worker process needs to seat players and score positions."""

    home: str
    stockfish: str | None = None


class LabelChunk(ArenaModel):
    """A slice of positions to score with one teacher."""

    fens: tuple[str, ...]
    teacher: TeacherKind
    depth: int


class _Worker:
    """Per-process engine factory and heuristic teacher.

    Stockfish processes are always opened and closed within one task: python-chess
    keeps a non-daemon thread per open engine, and an engine left open at exit
    would stop the worker process from ever shutting down.
    """

    def __init__(self, config: WorkerConfig) -> None:
        """Build the factory from the config."""
        self.config = config
        self.factory = EngineFactory(
            PackagedLadders(), FileModelStore(Path(config.home)), config.stockfish
        )
        self._heuristic: HeuristicTeacher | None = None

    def scorer(self) -> StockfishScorer:
        """Start a Stockfish scorer; the caller must close it."""
        if self.config.stockfish is None:
            msg = "the stockfish teacher needs a Stockfish binary"
            raise EngineUnavailableError(msg)
        return StockfishScorer(self.config.stockfish)

    def heuristic(self) -> HeuristicTeacher:
        """Return this worker's heuristic teacher."""
        if self._heuristic is None:
            self._heuristic = HeuristicTeacher()
        return self._heuristic


_STATE: dict[str, _Worker] = {}


def _init_worker(config: WorkerConfig) -> None:
    """Pool initializer: build this process's worker state."""
    _STATE["worker"] = _Worker(config)


def _worker() -> _Worker:
    """Return the state built by :func:`_init_worker`."""
    return _STATE["worker"]


def play_assignment(worker: _Worker, assignment: GameAssignment) -> GameRecord:
    """Seat both players and play one game."""
    started = datetime.now(UTC)
    white = worker.factory.open(
        assignment.white, assignment.seed * 2, _cap(assignment, assignment.white.kind)
    )
    black = worker.factory.open(
        assignment.black, assignment.seed * 2 + 1, _cap(assignment, assignment.black.kind)
    )
    try:
        return play_game(assignment, white, black, started, datetime.now(UTC))
    finally:
        white.close()
        black.close()


def _cap(assignment: GameAssignment, kind: PlayerKind) -> float | None:
    """Return the per-move cap for model players."""
    return assignment.model_max_seconds if kind is PlayerKind.MODEL else None


def data_games(worker: _Worker, task: DataGameTask) -> tuple[GamePositions, ...]:
    """Play one data task."""
    path = worker.config.stockfish
    return play_data_games(task, StockfishOpener(path) if path else None)


def label_chunk(worker: _Worker, chunk: LabelChunk) -> tuple[float, ...]:
    """Score one slice of positions."""
    if chunk.teacher is TeacherKind.STOCKFISH:
        scorer = worker.scorer()
        try:
            return tuple(scorer.score(fen, chunk.depth) for fen in chunk.fens)
        finally:
            scorer.close()
    teacher = worker.heuristic()
    return tuple(teacher.score(fen) for fen in chunk.fens)


def _pooled_game(assignment: GameAssignment) -> GameRecord:
    """Process-pool entry point for one game."""
    return play_assignment(_worker(), assignment)


def _pooled_data(task: DataGameTask) -> tuple[GamePositions, ...]:
    """Process-pool entry point for one data task."""
    return data_games(_worker(), task)


def _pooled_labels(chunk: LabelChunk) -> tuple[float, ...]:
    """Process-pool entry point for one labeling chunk."""
    return label_chunk(_worker(), chunk)


def _chunks(fens: tuple[str, ...], teacher: TeacherKind, depth: int) -> list[LabelChunk]:
    """Split positions into labeling chunks."""
    return [
        LabelChunk(fens=fens[start : start + _LABEL_CHUNK], teacher=teacher, depth=depth)
        for start in range(0, len(fens), _LABEL_CHUNK)
    ]


def _flatten[T](batches: Iterable[tuple[T, ...]]) -> tuple[T, ...]:
    """Concatenate result tuples in order."""
    return tuple(item for batch in batches for item in batch)


class ProcessPool:
    """One spawn-based process pool shared by games, data generation, and labeling."""

    def __init__(self, config: WorkerConfig, workers: int) -> None:
        """Start a pool of ``workers`` processes (each builds its own worker state)."""
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config,),
        )

    def play(self, assignments: tuple[GameAssignment, ...]) -> Generator[GameRecord]:
        """Yield games as they finish; closing the generator cancels queued games."""
        futures: list[Future[GameRecord]] = [
            self._executor.submit(_pooled_game, assignment) for assignment in assignments
        ]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                _ = future.cancel()

    def generate(self, tasks: tuple[DataGameTask, ...]) -> tuple[GamePositions, ...]:
        """Play every data task across the pool, keeping task order."""
        return _flatten(self._executor.map(_pooled_data, tasks))

    def label(self, fens: tuple[str, ...], teacher: TeacherKind, depth: int) -> tuple[float, ...]:
        """Score positions across the pool, keeping their order."""
        return _flatten(self._executor.map(_pooled_labels, _chunks(fens, teacher, depth)))

    def shutdown(self) -> None:
        """Stop the worker processes."""
        self._executor.shutdown(wait=False, cancel_futures=True)


class InlinePool:
    """The same operations in the current process, one at a time."""

    def __init__(self, config: WorkerConfig) -> None:
        """Build the worker state locally."""
        self._worker = _Worker(config)

    def play(self, assignments: tuple[GameAssignment, ...]) -> Generator[GameRecord]:
        """Yield games in order."""
        for assignment in assignments:
            yield play_assignment(self._worker, assignment)

    def generate(self, tasks: tuple[DataGameTask, ...]) -> tuple[GamePositions, ...]:
        """Play every data task in order."""
        return _flatten(data_games(self._worker, task) for task in tasks)

    def label(self, fens: tuple[str, ...], teacher: TeacherKind, depth: int) -> tuple[float, ...]:
        """Score positions in order."""
        return _flatten(label_chunk(self._worker, chunk) for chunk in _chunks(fens, teacher, depth))

    def shutdown(self) -> None:
        """Nothing to stop: every engine is closed by the task that opened it."""
