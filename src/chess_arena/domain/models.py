"""Model cards: identity, lineage, recipe, and training summary of a trained model."""

from typing import Literal

from pydantic import AwareDatetime, Field

from chess_arena.domain.contract import ArenaModel, LongText, ShortText, Slug
from chess_arena.domain.recipes import TrainingRecipe


class TrainingSummary(ArenaModel):
    """What the trainer actually did for one model."""

    positions: int = Field(ge=0)
    games: int = Field(ge=0)
    rounds: int = Field(ge=0)
    epochs_run: int = Field(ge=0)
    train_loss: float | None = None
    validation_loss: float | None = None
    seconds: float = Field(ge=0)
    teacher: str = "none"
    stopped_early: bool = False
    initialized_from: Slug | None = None


class ModelCard(ArenaModel):
    """A trained (or hand-written baseline) model the arena can seat as a player."""

    model_id: Slug
    format: Literal["valuenet-v1"] = "valuenet-v1"
    label: ShortText
    recipe: TrainingRecipe
    recipe_digest: str = Field(min_length=64, max_length=64)
    parent_id: Slug | None = None
    created_at: AwareDatetime
    training: TrainingSummary | None = None
    tags: tuple[Slug, ...] = ()
    notes: LongText = ""


class ModelUpload(ArenaModel):
    """A model card plus its weights as base64-encoded ``.npz`` bytes."""

    card: ModelCard
    weights_npz_base64: str = Field(min_length=1, max_length=64_000_000)


class ModelSummary(ArenaModel):
    """A model card together with its best ladder result, for listings."""

    card: ModelCard
    best_ladder_score: float | None = None
    best_elo_estimate: float | None = None
    highest_beaten: Slug | None = None
    reports: int = Field(default=0, ge=0)
