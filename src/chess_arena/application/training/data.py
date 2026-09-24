"""Plan and play training games, then sample positions from them.

Data games are split into tasks of at most 25 games so a process pool can play
them in parallel. Every task carries its own seed, so the same recipe and seed
produce the same games regardless of how many workers run them.
"""

import chess
import numpy as np

from chess_arena.application.engines.bots import ExploringEngine, GreedyBot, MinimaxBot
from chess_arena.application.engines.model_player import ModelPlayer
from chess_arena.application.engines.positions import finished
from chess_arena.application.learning.serialization import from_blob
from chess_arena.application.ports.engines import EnginePort, MoveRequest
from chess_arena.application.ports.training import (
    DataGameKind,
    DataGameTask,
    GamePositions,
    NetworkBlob,
    StockfishOpenerPort,
)
from chess_arena.domain.errors import EngineUnavailableError
from chess_arena.domain.games import GameResult
from chess_arena.domain.recipes import SearchSpec, TrainingRecipe

GAMES_PER_TASK = 25
_FIRST_SAMPLED_PLY = 2
_RANDOM_FLOOR = 0.5
_STOCKFISH_ELO_RANGE = (1320, 2400)
_STOCKFISH_MOVE_MS = 5
_SELF_PLAY_SEARCH = SearchSpec(
    depth=1, quiescence_depth=2, material_blend=0.0, max_seconds_per_move=0.2
)


def plan_tasks(
    recipe: TrainingRecipe, round_index: int, network: NetworkBlob | None
) -> tuple[DataGameTask, ...]:
    """Split the recipe's data games into seeded tasks for one round."""
    data = recipe.data
    keep_all = recipe.labels.td_lambda is not None
    sources = (
        (DataGameKind.RANDOM, data.random_games),
        (DataGameKind.BOTS, data.bot_games),
        (DataGameKind.STOCKFISH, data.stockfish_games),
        (DataGameKind.SELF_PLAY, data.self_play_games),
    )
    tasks: list[DataGameTask] = []
    for kind_index, (kind, games) in enumerate(sources):
        for chunk, start in enumerate(range(0, games, GAMES_PER_TASK)):
            self_play = kind is DataGameKind.SELF_PLAY
            tasks.append(
                DataGameTask(
                    kind=kind,
                    games=min(GAMES_PER_TASK, games - start),
                    seed=recipe.seed * 1_000_003
                    + round_index * 10_007
                    + kind_index * 1_009
                    + chunk,
                    exploration=data.exploration,
                    positions_per_game=data.positions_per_game,
                    keep_all_positions=keep_all and self_play,
                    network=network if self_play else None,
                    eval_scale_cp=recipe.labels.eval_scale_cp,
                )
            )
    return tuple(tasks)


def _stockfish_pair(
    task: DataGameTask, rng: np.random.Generator, stockfish: StockfishOpenerPort | None
) -> list[EnginePort]:
    """Open two Stockfish players at random Elos, shared by every game of the task.

    Starting a Stockfish process costs far more than a fast game, so one pair
    serves the whole task and the task's seed picks its strengths.
    """
    if task.kind is not DataGameKind.STOCKFISH:
        return []
    if stockfish is None:
        msg = "stockfish data games need a Stockfish binary"
        raise EngineUnavailableError(msg)
    low, high = _STOCKFISH_ELO_RANGE
    return [stockfish.open(int(rng.integers(low, high + 1)), _STOCKFISH_MOVE_MS) for _ in range(2)]


def _engine(
    task: DataGameTask,
    rng: np.random.Generator,
    shared: EnginePort | None,
    player: ModelPlayer | None,
) -> EnginePort:
    """Return one exploring engine for ``task``'s kind of game."""
    match task.kind:
        case DataGameKind.RANDOM:
            return ExploringEngine(GreedyBot(rng), max(_RANDOM_FLOOR, task.exploration), rng)
        case DataGameKind.BOTS:
            inner: EnginePort = MinimaxBot(1, rng, noise_cp=30.0, seconds=0.5)
        case DataGameKind.STOCKFISH:
            if shared is None:
                msg = "stockfish data games need an open Stockfish player"
                raise EngineUnavailableError(msg)
            inner = shared
        case DataGameKind.SELF_PLAY:
            if player is None:
                msg = "self-play games need a network"
                raise ValueError(msg)
            inner = player
    return ExploringEngine(inner, task.exploration, rng)


def _play(white: EnginePort, black: EnginePort, max_plies: int) -> tuple[list[str], float]:
    """Play one game; return the FEN before every move and White's outcome."""
    board = chess.Board()
    fens: list[str] = []
    moves: list[str] = []
    while len(moves) < max_plies:
        outcome = finished(board)
        if outcome is not None:
            result = outcome[0]
            white_points = 1.0 if result is GameResult.WHITE_WINS else 0.0
            black_points = 1.0 if result is GameResult.BLACK_WINS else 0.0
            return fens, white_points - black_points
        fens.append(board.fen())
        engine = white if board.turn == chess.WHITE else black
        choice = engine.choose(MoveRequest(moves=tuple(moves)))
        _ = board.push_uci(choice.uci)
        moves.append(choice.uci)
    return fens, 0.0


def _sample(fens: list[str], count: int, rng: np.random.Generator) -> list[str]:
    """Return up to ``count`` positions after the first plies, in game order."""
    eligible = fens[_FIRST_SAMPLED_PLY:]
    if len(eligible) <= count:
        return eligible
    chosen = np.sort(rng.choice(len(eligible), size=count, replace=False))
    return [eligible[int(index)] for index in chosen]


def play_data_games(
    task: DataGameTask, stockfish: StockfishOpenerPort | None
) -> tuple[GamePositions, ...]:
    """Play every game of ``task`` and return the sampled positions."""
    rng = np.random.default_rng(task.seed)
    player = None
    if task.network is not None:
        network = from_blob(task.network)
        player = ModelPlayer(network, _SELF_PLAY_SEARCH, task.eval_scale_cp, rng)
    games: list[GamePositions] = []
    shared = _stockfish_pair(task, rng, stockfish)
    try:
        for index in range(task.games):
            pair = (shared[index % 2], shared[(index + 1) % 2]) if shared else (None, None)
            white = _engine(task, rng, pair[0], player)
            black = _engine(task, rng, pair[1], player)
            fens, outcome = _play(white, black, task.max_plies)
            kept = fens if task.keep_all_positions else _sample(fens, task.positions_per_game, rng)
            games.append(
                GamePositions(
                    fens=tuple(kept), outcome_white=outcome, complete=task.keep_all_positions
                )
            )
    finally:
        for engine in shared:
            engine.close()
    return tuple(games)
