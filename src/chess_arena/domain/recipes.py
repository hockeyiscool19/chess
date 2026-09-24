"""Bounded training recipes for ``valuenet-v1`` models.

A recipe is data, not code: every knob has a hard range so an automated
researcher can propose one without being able to run arbitrary programs or
unbounded compute. The trainer turns a recipe into positions, labels, and a
fitted value network; the model then plays through a small alpha-beta search.
"""

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from chess_arena.domain.contract import ArenaModel

MAX_POSITIONS = 400_000
MAX_TOTAL_POSITIONS = 800_000
MAX_HIDDEN = 512
MIN_HIDDEN = 8


class Activation(StrEnum):
    """Hidden-layer nonlinearity."""

    CRELU = "crelu"
    TANH = "tanh"


class TeacherKind(StrEnum):
    """Where supervised position values come from."""

    NONE = "none"
    HEURISTIC = "heuristic"
    STOCKFISH = "stockfish"


class InitMode(StrEnum):
    """Start from fresh random weights or fine-tune the parent model."""

    SCRATCH = "scratch"
    PARENT = "parent"


class NetworkSpec(ArenaModel):
    """A 768-input multilayer perceptron with one scalar output."""

    hidden: tuple[int, ...] = Field(
        default=(128, 32),
        min_length=1,
        max_length=3,
        description="Hidden layer widths after the 768 piece-square inputs, each 8..512.",
    )
    activation: Activation = Field(
        default=Activation.CRELU,
        description="crelu clips to [0, 1] (fast, NNUE-style); tanh is smooth.",
    )

    @field_validator("hidden")
    @classmethod
    def widths_in_range(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject layer widths outside 8..512."""
        if any(width < MIN_HIDDEN or width > MAX_HIDDEN for width in value):
            msg = f"hidden widths must be between {MIN_HIDDEN} and {MAX_HIDDEN}"
            raise ValueError(msg)
        return value


class DataSpec(ArenaModel):
    """How many games of each kind to play for training positions."""

    random_games: int = Field(
        default=0, ge=0, le=4000, description="Near-random games: diverse, unrealistic positions."
    )
    bot_games: int = Field(
        default=200, ge=0, le=4000, description="Games between noisy depth-1 minimax bots."
    )
    stockfish_games: int = Field(
        default=0,
        ge=0,
        le=4000,
        description="Strength-limited Stockfish self-play games: realistic positions.",
    )
    self_play_games: int = Field(
        default=0,
        ge=0,
        le=4000,
        description="Games of the current network against itself (reinforcement learning).",
    )
    positions_per_game: int = Field(default=10, ge=1, le=200)
    exploration: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Chance of a random move in data games."
    )

    @property
    def total_games(self) -> int:
        """Return the number of data games across all sources."""
        return self.random_games + self.bot_games + self.stockfish_games + self.self_play_games

    @model_validator(mode="after")
    def bounded_positions(self) -> Self:
        """Cap the total sampled positions."""
        if self.total_games * self.positions_per_game > MAX_POSITIONS:
            msg = f"games x positions_per_game must not exceed {MAX_POSITIONS}"
            raise ValueError(msg)
        return self


class LabelSpec(ArenaModel):
    """How each position's target value in [-1, 1] is computed."""

    teacher: TeacherKind = TeacherKind.STOCKFISH
    teacher_depth: int = Field(default=8, ge=1, le=14, description="Stockfish search depth.")
    outcome_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="target = w * game_outcome + (1 - w) * teacher_value.",
    )
    td_lambda: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="If set, self-play outcomes become TD(lambda) returns of the network.",
    )
    eval_scale_cp: float = Field(
        default=400.0, ge=100.0, le=1500.0, description="teacher_value = tanh(cp / scale)."
    )

    @model_validator(mode="after")
    def teacherless_uses_outcomes(self) -> Self:
        """Without a teacher the only signal is the game outcome."""
        if self.teacher is TeacherKind.NONE and self.outcome_weight != 1.0:
            msg = "teacher 'none' requires outcome_weight 1.0"
            raise ValueError(msg)
        return self


class OptimizerSpec(ArenaModel):
    """Adam on mean squared error with early stopping on a validation split."""

    epochs: int = Field(default=20, ge=1, le=200)
    batch_size: int = Field(default=256, ge=16, le=4096)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=0.1)
    weight_decay: float = Field(default=0.0, ge=0.0, le=0.1)
    validation_fraction: float = Field(default=0.1, ge=0.0, le=0.5)
    patience: int = Field(default=5, ge=1, le=50)


class SearchSpec(ArenaModel):
    """How the model chooses moves at play time."""

    depth: int = Field(default=2, ge=1, le=4, description="Full-width alpha-beta depth.")
    quiescence_depth: int = Field(default=4, ge=0, le=8, description="Capture-only extension.")
    material_blend: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 = network evaluation only, 1 = hand-written material evaluation only.",
    )
    max_seconds_per_move: float = Field(default=1.0, ge=0.05, le=10.0)


class TrainingRecipe(ArenaModel):
    """Everything needed to reproduce one model from its parent."""

    format: Literal["valuenet-v1"] = "valuenet-v1"
    init: InitMode = InitMode.PARENT
    network: NetworkSpec = NetworkSpec()
    data: DataSpec = DataSpec()
    labels: LabelSpec = LabelSpec()
    optimizer: OptimizerSpec = OptimizerSpec()
    rounds: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Repeat play-label-fit; later rounds replay self-play with the new network.",
    )
    search: SearchSpec = SearchSpec()
    seed: int = Field(default=1, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def td_needs_self_play(self) -> Self:
        """TD(lambda) needs self-play games; all rounds together stay bounded."""
        if self.labels.td_lambda is not None and self.data.self_play_games == 0:
            msg = "labels.td_lambda requires data.self_play_games > 0"
            raise ValueError(msg)
        positions = self.data.total_games * self.data.positions_per_game * self.rounds
        if positions > MAX_TOTAL_POSITIONS:
            msg = f"games x positions_per_game x rounds must not exceed {MAX_TOTAL_POSITIONS}"
            raise ValueError(msg)
        return self

    @property
    def trains(self) -> bool:
        """Return whether this recipe generates any training data."""
        return self.data.total_games > 0

    @property
    def uses_stockfish(self) -> bool:
        """Return whether training needs a Stockfish binary."""
        return self.labels.teacher is TeacherKind.STOCKFISH or self.data.stockfish_games > 0

    def digest(self) -> str:
        """Return a SHA-256 over the canonical JSON of this recipe."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def baseline_recipe() -> TrainingRecipe:
    """Return the untrained baseline: hand-written evaluation at depth 2."""
    return TrainingRecipe(
        init=InitMode.SCRATCH,
        data=DataSpec(bot_games=0),
        search=SearchSpec(depth=2, material_blend=1.0),
    )
