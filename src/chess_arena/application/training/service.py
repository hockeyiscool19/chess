"""Train one model from a recipe: play data games, label positions, fit, and save.

Each round plays the recipe's data games (self-play games use the network as it
stands at the start of the round), labels the sampled positions, adds them to
the replay buffer, and fits the network on the whole buffer. A wall-clock budget
stops training between rounds; the card records whether that happened.
"""

import chess
import numpy as np
from numpy.typing import NDArray
from pydantic import AwareDatetime, Field

from chess_arena.application.learning.features import feature_indices, index_matrix
from chess_arena.application.learning.fit import FitReport, fit
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.learning.serialization import to_blob
from chess_arena.application.ports.runtime import ClockPort
from chess_arena.application.ports.training import (
    DataGeneratorPort,
    GamePositions,
    ModelSinkPort,
    TeacherPort,
)
from chess_arena.application.training.data import plan_tasks
from chess_arena.application.training.labels import (
    blend_targets,
    mover_outcome,
    sample_indices,
    td_returns,
)
from chess_arena.domain.contract import ArenaModel, LongText, ShortText, Slug
from chess_arena.domain.errors import InvalidRecipeError
from chess_arena.domain.models import ModelCard, TrainingSummary
from chess_arena.domain.recipes import InitMode, TeacherKind, TrainingRecipe


class TrainRequest(ArenaModel):
    """Train ``recipe`` into a new model, optionally from a parent model."""

    model_id: Slug
    label: ShortText
    recipe: TrainingRecipe
    parent_id: Slug | None = None
    tags: tuple[Slug, ...] = ()
    notes: LongText = ""
    max_seconds: float = Field(default=3600.0, gt=0, le=86_400)


class TrainingPorts:
    """Data generation, labeling, and model persistence."""

    def __init__(self, games: DataGeneratorPort, teacher: TeacherPort, sink: ModelSinkPort) -> None:
        """Bind the three ports training needs."""
        self.games = games
        self.teacher = teacher
        self.sink = sink


class _Buffer:
    """Accumulated feature rows and targets across rounds."""

    def __init__(self) -> None:
        """Start empty."""
        self.rows: list[list[int]] = []
        self.targets: list[float] = []

    def arrays(self) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
        """Return the buffer as training arrays."""
        return index_matrix(self.rows), np.asarray(self.targets, dtype=np.float32)


class TrainingService:
    """Execute a recipe end to end."""

    def __init__(self, ports: TrainingPorts, clock: ClockPort) -> None:
        """Bind ports and the clock used for timestamps and the time budget."""
        self._ports = ports
        self._clock = clock

    def train(self, request: TrainRequest, parent: ValueNetwork | None) -> ModelCard:
        """Train, save through the sink, and return the new model card."""
        recipe = request.recipe
        started: AwareDatetime = self._clock.now()
        rng = np.random.default_rng([recipe.seed, 17])
        network = self._initial(recipe, parent, rng)
        buffer = _Buffer()
        report = FitReport(epochs_run=0)
        rounds = 0
        stopped = False
        games = 0
        for round_index in range(recipe.rounds if recipe.trains else 0):
            if self._elapsed(started) > request.max_seconds:
                stopped = True
                break
            blob = to_blob(network) if recipe.data.self_play_games else None
            played = self._ports.games.generate(plan_tasks(recipe, round_index, blob))
            games += len(played)
            self._add(buffer, played, recipe, network, rng)
            indices, targets = buffer.arrays()
            report = fit(network, indices, targets, recipe.optimizer, rng)
            rounds += 1
        card = ModelCard(
            model_id=request.model_id,
            label=request.label,
            recipe=recipe,
            recipe_digest=recipe.digest(),
            parent_id=request.parent_id,
            created_at=self._clock.now(),
            training=TrainingSummary(
                positions=len(buffer.targets),
                games=games,
                rounds=rounds,
                epochs_run=report.epochs_run,
                train_loss=report.train_loss,
                validation_loss=report.validation_loss,
                seconds=round(self._elapsed(started), 2),
                teacher=recipe.labels.teacher.value if recipe.trains else "none",
                stopped_early=stopped or report.stopped_early,
                initialized_from=request.parent_id if self._from_parent(recipe, parent) else None,
            ),
            tags=request.tags,
            notes=request.notes,
        )
        self._ports.sink.save(card, network)
        return card

    def _from_parent(self, recipe: TrainingRecipe, parent: ValueNetwork | None) -> bool:
        """Return whether training starts from the parent's weights."""
        return recipe.init is InitMode.PARENT and parent is not None

    def _initial(
        self, recipe: TrainingRecipe, parent: ValueNetwork | None, rng: np.random.Generator
    ) -> ValueNetwork:
        """Return the parent's weights (same architecture required) or fresh weights."""
        if not self._from_parent(recipe, parent) or parent is None:
            return ValueNetwork.initialize(recipe.network, rng)
        if parent.spec != recipe.network:
            msg = (
                f"init 'parent' needs the parent's architecture {parent.spec.model_dump_json()}; "
                "use init 'scratch' to change layers"
            )
            raise InvalidRecipeError(msg)
        return parent.copy()

    def _add(
        self,
        buffer: _Buffer,
        played: tuple[GamePositions, ...],
        recipe: TrainingRecipe,
        network: ValueNetwork,
        rng: np.random.Generator,
    ) -> None:
        """Label the positions of ``played`` and append them to ``buffer``."""
        fens: list[str] = []
        outcomes: list[float] = []
        td = recipe.labels.td_lambda
        for game in played:
            if td is not None and game.complete:
                values = td_returns(game, network, td)
                keep = sample_indices(len(game.fens), recipe.data.positions_per_game, rng)
                fens.extend(game.fens[index] for index in keep)
                outcomes.extend(values[index] for index in keep)
            else:
                fens.extend(game.fens)
                outcomes.extend(mover_outcome(fen, game.outcome_white) for fen in game.fens)
        teacher = recipe.labels.teacher
        scores = None
        if teacher is not TeacherKind.NONE and fens:
            scores = list(
                self._ports.teacher.label(tuple(fens), teacher, recipe.labels.teacher_depth)
            )
        targets = blend_targets(outcomes, scores, recipe.labels)
        buffer.rows.extend(feature_indices(chess.Board(fen)) for fen in fens)
        buffer.targets.extend(float(value) for value in targets)

    def _elapsed(self, started: AwareDatetime) -> float:
        """Return seconds since ``started``."""
        return (self._clock.now() - started).total_seconds()
