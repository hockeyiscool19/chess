"""Model listing, detail, and upload (the institute publishes models here)."""

import base64
import binascii

from fastapi import APIRouter

from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.application.catalog import summarize
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.learning.serialization import from_npz_bytes
from chess_arena.domain.contract import ArenaModel
from chess_arena.domain.errors import InvalidRecipeError
from chess_arena.domain.ladder import LadderReport
from chess_arena.domain.models import ModelCard, ModelSummary, ModelUpload


class ModelDetail(ArenaModel):
    """A model's summary plus every finished ladder report."""

    summary: ModelSummary
    reports: tuple[LadderReport, ...] = ()


def _network(upload: ModelUpload) -> ValueNetwork:
    """Decode and check uploaded weights against the card's architecture."""
    try:
        raw = base64.b64decode(upload.weights_npz_base64, validate=True)
        arrays = from_npz_bytes(raw)
        network = ValueNetwork.from_arrays(arrays, upload.card.recipe.network.activation)
    except (binascii.Error, ValueError, KeyError, OSError) as exc:
        msg = f"weights are not a valid valuenet-v1 archive: {exc}"
        raise InvalidRecipeError(msg) from exc
    if network.spec != upload.card.recipe.network:
        msg = "uploaded weights do not match the recipe's network architecture"
        raise InvalidRecipeError(msg)
    if upload.card.recipe_digest != upload.card.recipe.digest():
        msg = "recipe_digest does not match the recipe"
        raise InvalidRecipeError(msg)
    return network


def models_router(arena: Arena) -> APIRouter:
    """Return routes for models."""
    router = APIRouter(prefix="/api", tags=["models"])

    @router.get("/models")
    def listing() -> list[ModelSummary]:
        """List every model with its best ladder result, oldest first."""
        return [summarize(card, arena.jobs.reports(card.model_id)) for card in arena.models.cards()]

    @router.get("/models/{model_id}")
    def detail(model_id: str) -> ModelDetail:
        """Return one model and its ladder reports."""
        card = arena.models.card(model_id)
        reports = arena.jobs.reports(model_id)
        return ModelDetail(summary=summarize(card, reports), reports=reports)

    @router.post("/models", status_code=201)
    def upload(request: ModelUpload) -> ModelCard:
        """Store a model; uploading the identical model again is a no-op."""
        arena.models.save(request.card, _network(request))
        return request.card

    return router
