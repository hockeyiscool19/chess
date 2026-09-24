from pathlib import Path

import numpy as np
import pytest

from chess_arena.adapters.outbound.engines.factory import EngineFactory
from chess_arena.adapters.outbound.engines.stockfish import find_stockfish
from chess_arena.adapters.outbound.ladders import PackagedLadders
from chess_arena.adapters.outbound.store.files import (
    DirectoryModelSink,
    FileGameStore,
    FileModelStore,
    read_model_folder,
)
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.domain.errors import ConflictError, EngineUnavailableError, NotFoundError
from chess_arena.domain.games import GameContext, GameKind, GameRecord
from chess_arena.domain.models import ModelCard
from chess_arena.domain.players import PlayerKind, PlayerRef
from chess_arena.domain.recipes import NetworkSpec, TrainingRecipe
from tests.helpers import StepClock


def _card(model_id: str = "m1", notes: str = "") -> ModelCard:
    recipe = TrainingRecipe(network=NetworkSpec(hidden=(8,)))
    return ModelCard(
        model_id=model_id,
        label=model_id,
        recipe=recipe,
        recipe_digest=recipe.digest(),
        created_at=StepClock().now(),
        notes=notes,
    )


def _network() -> ValueNetwork:
    return ValueNetwork.initialize(NetworkSpec(hidden=(8,)), np.random.default_rng(0))


def test_packaged_ladder_is_ordered_and_reaches_full_stockfish() -> None:
    ladder = PackagedLadders().ladder("ladder-v1")
    assert ladder.rungs[0].rung_id == "random"
    assert ladder.rungs[-1].rung_id == "stockfish-full"
    assert len(ladder.rungs) == 14
    with pytest.raises(NotFoundError):
        _ = PackagedLadders().ladder("nope")


def test_model_store_round_trip_and_conflicts(tmp_path: Path) -> None:
    store = FileModelStore(tmp_path)
    card, network = _card(), _network()
    store.save(card, network)
    store.save(card, network)
    assert store.card("m1") == card
    np.testing.assert_array_equal(store.network("m1").weights[0], network.weights[0])
    with pytest.raises(ConflictError):
        store.save(_card(notes="different"), network)
    with pytest.raises(NotFoundError):
        _ = store.card("missing")
    assert [item.model_id for item in store.cards()] == ["m1"]


def test_directory_sink_is_readable(tmp_path: Path) -> None:
    DirectoryModelSink(tmp_path / "out").save(_card("m2"), _network())
    card, network = read_model_folder(tmp_path / "out")
    assert card.model_id == "m2"
    assert network.spec == NetworkSpec(hidden=(8,))


def test_game_store_filters_by_kind(tmp_path: Path) -> None:
    store = FileGameStore(tmp_path)
    clock = StepClock()
    for index, kind in enumerate((GameKind.HUMAN, GameKind.LADDER, GameKind.LADDER)):
        store.save(
            GameRecord(
                game_id=f"g{index}",
                white=PlayerRef(kind=PlayerKind.ENGINE, player_id="random"),
                black=PlayerRef(kind=PlayerKind.ENGINE, player_id="greedy"),
                context=GameContext(kind=kind),
                created_at=clock.now(),
            )
        )
    assert len(store.recent(GameKind.LADDER, 10)) == 2
    assert len(store.recent(None, 2)) == 2
    with pytest.raises(NotFoundError):
        _ = store.get("missing")


def test_factory_reports_missing_stockfish(tmp_path: Path) -> None:
    factory = EngineFactory(PackagedLadders(), FileModelStore(tmp_path), None)
    opponents = {item.player.player_id: item for item in factory.opponents()}
    assert opponents["random"].available
    assert not opponents["stockfish-1320"].available
    with pytest.raises(EngineUnavailableError):
        _ = factory.open(PlayerRef(kind=PlayerKind.ENGINE, player_id="stockfish-1320"), 0)
    assert find_stockfish("definitely-missing-binary") is None
