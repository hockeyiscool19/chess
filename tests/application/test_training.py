import chess
import numpy as np
import pytest

from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.ports.training import (
    DataGameKind,
    DataGameTask,
    GamePositions,
)
from chess_arena.application.training.data import GAMES_PER_TASK, plan_tasks, play_data_games
from chess_arena.application.training.labels import blend_targets, mover_outcome, td_returns
from chess_arena.application.training.service import TrainingPorts, TrainingService, TrainRequest
from chess_arena.domain.errors import InvalidRecipeError
from chess_arena.domain.models import ModelCard
from chess_arena.domain.recipes import (
    DataSpec,
    InitMode,
    LabelSpec,
    NetworkSpec,
    OptimizerSpec,
    TeacherKind,
    TrainingRecipe,
    baseline_recipe,
)
from tests.helpers import StepClock


class FakeGames:
    """Returns a few scripted games per task."""

    def __init__(self) -> None:
        self.tasks: list[DataGameTask] = []

    def generate(self, tasks: tuple[DataGameTask, ...]) -> tuple[GamePositions, ...]:
        self.tasks.extend(tasks)
        board = chess.Board()
        fens: list[str] = []
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"):
            fens.append(board.fen())
            _ = board.push_uci(uci)
        return tuple(
            GamePositions(fens=tuple(fens), outcome_white=1.0, complete=task.keep_all_positions)
            for task in tasks
            for _ in range(task.games)
        )


class FakeTeacher:
    """Scores every position +100 centipawns for the side to move."""

    def __init__(self) -> None:
        self.calls = 0

    def label(self, fens: tuple[str, ...], teacher: TeacherKind, depth: int) -> tuple[float, ...]:
        del teacher, depth
        self.calls += 1
        return tuple(100.0 for _ in fens)


class MemorySink:
    """Keeps the saved card and network."""

    def __init__(self) -> None:
        self.saved: list[tuple[ModelCard, ValueNetwork]] = []

    def save(self, card: ModelCard, network: ValueNetwork) -> None:
        self.saved.append((card, network))


def _service() -> tuple[TrainingService, FakeGames, FakeTeacher, MemorySink]:
    games, teacher, sink = FakeGames(), FakeTeacher(), MemorySink()
    return TrainingService(TrainingPorts(games, teacher, sink), StepClock()), games, teacher, sink


def test_plan_tasks_splits_games_with_distinct_seeds() -> None:
    recipe = TrainingRecipe(data=DataSpec(bot_games=60, random_games=10))
    tasks = plan_tasks(recipe, 0, None)
    assert sum(task.games for task in tasks) == 70
    assert all(task.games <= GAMES_PER_TASK for task in tasks)
    assert len({task.seed for task in tasks}) == len(tasks)
    assert {task.kind for task in tasks} == {DataGameKind.RANDOM, DataGameKind.BOTS}
    later = plan_tasks(recipe, 1, None)
    assert {task.seed for task in tasks}.isdisjoint({task.seed for task in later})


def test_td_returns_interpolate_between_bootstrap_and_outcome() -> None:
    network = ValueNetwork.initialize(NetworkSpec(hidden=(8,)), np.random.default_rng(0))
    for weight in network.weights:
        weight[...] = 0.0
    board = chess.Board()
    fens = [board.fen()]
    _ = board.push_uci("e2e4")
    fens.append(board.fen())
    game = GamePositions(fens=tuple(fens), outcome_white=1.0, complete=True)
    assert td_returns(game, network, 1.0) == pytest.approx([1.0, -1.0])
    assert td_returns(game, network, 0.0) == pytest.approx([0.0, -1.0])


def test_mover_outcome_and_blending() -> None:
    white = chess.Board().fen()
    assert mover_outcome(white, 1.0) == 1.0
    black = chess.Board("8/8/8/8/8/8/8/K6k b - - 0 1").fen()
    assert mover_outcome(black, 1.0) == -1.0
    targets = blend_targets([1.0], [0.0], LabelSpec(outcome_weight=0.25))
    assert targets[0] == pytest.approx(0.25)


def test_training_runs_rounds_and_saves_a_card() -> None:
    service, games, teacher, sink = _service()
    recipe = TrainingRecipe(
        network=NetworkSpec(hidden=(8,)),
        data=DataSpec(bot_games=2, self_play_games=2, positions_per_game=4),
        labels=LabelSpec(td_lambda=0.8),
        optimizer=OptimizerSpec(epochs=2, validation_fraction=0.0),
        rounds=2,
    )
    card = service.train(TrainRequest(model_id="m-1", label="M1", recipe=recipe), None)
    assert card.training is not None
    assert card.training.rounds == 2
    assert card.training.positions > 0
    assert teacher.calls == 2
    assert any(task.keep_all_positions for task in games.tasks)
    assert sink.saved
    assert sink.saved[0][0] == card
    assert card.recipe_digest == recipe.digest()


def test_baseline_recipe_skips_training() -> None:
    service, games, _, sink = _service()
    card = service.train(
        TrainRequest(model_id="base", label="Base", recipe=baseline_recipe()), None
    )
    assert card.training is not None
    assert card.training.positions == 0
    assert not games.tasks
    assert sink.saved


def test_parent_init_requires_the_same_architecture() -> None:
    service, *_ = _service()
    parent = ValueNetwork.initialize(NetworkSpec(hidden=(16,)), np.random.default_rng(0))
    recipe = TrainingRecipe(network=NetworkSpec(hidden=(8,)), init=InitMode.PARENT)
    with pytest.raises(InvalidRecipeError):
        _ = service.train(
            TrainRequest(model_id="c", label="C", recipe=recipe, parent_id="p"), parent
        )


def test_bot_data_games_are_reproducible() -> None:
    task = DataGameTask(
        kind=DataGameKind.RANDOM,
        games=2,
        seed=7,
        exploration=0.5,
        positions_per_game=3,
        max_plies=40,
    )
    first = play_data_games(task, None)
    assert first == play_data_games(task, None)
    assert all(len(game.fens) <= 3 for game in first)
