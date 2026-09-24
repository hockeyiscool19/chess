import pytest
from pydantic import ValidationError

from chess_arena.domain.recipes import (
    DataSpec,
    LabelSpec,
    NetworkSpec,
    TeacherKind,
    TrainingRecipe,
    baseline_recipe,
)


def test_baseline_recipe_does_not_train_and_uses_the_hand_written_eval() -> None:
    recipe = baseline_recipe()
    assert not recipe.trains
    assert recipe.search.material_blend == 1.0


def test_recipe_digest_is_stable_and_sensitive() -> None:
    first = TrainingRecipe()
    assert first.digest() == TrainingRecipe().digest()
    assert first.digest() != TrainingRecipe(seed=2).digest()


def test_hidden_widths_are_bounded() -> None:
    with pytest.raises(ValidationError):
        _ = NetworkSpec(hidden=(4,))
    with pytest.raises(ValidationError):
        _ = NetworkSpec(hidden=(64, 64, 64, 64))


def test_positions_are_bounded() -> None:
    with pytest.raises(ValidationError):
        _ = DataSpec(bot_games=4000, positions_per_game=200)
    with pytest.raises(ValidationError):
        _ = TrainingRecipe(data=DataSpec(bot_games=2000, positions_per_game=100), rounds=8)


def test_td_lambda_requires_self_play() -> None:
    with pytest.raises(ValidationError):
        _ = TrainingRecipe(labels=LabelSpec(td_lambda=0.7))
    recipe = TrainingRecipe(
        data=DataSpec(bot_games=0, self_play_games=10), labels=LabelSpec(td_lambda=0.7)
    )
    assert recipe.labels.td_lambda == 0.7


def test_teacherless_labels_must_use_outcomes_only() -> None:
    with pytest.raises(ValidationError):
        _ = LabelSpec(teacher=TeacherKind.NONE, outcome_weight=0.5)
    assert LabelSpec(teacher=TeacherKind.NONE, outcome_weight=1.0).outcome_weight == 1.0


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = TrainingRecipe.model_validate({"format": "valuenet-v1", "code": "import os"})


def test_uses_stockfish_reflects_teacher_and_data() -> None:
    assert TrainingRecipe().uses_stockfish
    heuristic = TrainingRecipe(labels=LabelSpec(teacher=TeacherKind.HEURISTIC))
    assert not heuristic.uses_stockfish
