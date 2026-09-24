from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from chess_arena.adapters.outbound.store.files import FileGameStore, FileJobStore, FileModelStore
from chess_arena.application.ladder.service import LadderService, LadderStores
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.ports.runtime import GameAssignment
from chess_arena.domain.errors import NotFoundError
from chess_arena.domain.games import GameRecord, GameResult, Termination
from chess_arena.domain.jobs import JobStatus, LadderRunRequest
from chess_arena.domain.ladder import LadderSpec, Rung
from chess_arena.domain.models import ModelCard
from chess_arena.domain.players import EngineFamily, EngineSpec, PlayerKind
from chess_arena.domain.recipes import NetworkSpec, TrainingRecipe
from tests.helpers import StepClock


class Ladders:
    def __init__(self, spec: LadderSpec) -> None:
        self.spec = spec

    def ladder(self, ladder_id: str) -> LadderSpec:
        if ladder_id != self.spec.ladder_id:
            raise NotFoundError(ladder_id)
        return self.spec

    def ladders(self) -> tuple[LadderSpec, ...]:
        return (self.spec,)


class ScriptedRunner:
    """The candidate wins against every rung id in ``beats`` and loses otherwise."""

    def __init__(self, beats: set[str]) -> None:
        self.beats = beats
        self.requested = 0
        self.closed = 0

    def play(self, assignments: tuple[GameAssignment, ...]) -> Generator[GameRecord]:
        try:
            for assignment in assignments:
                self.requested += 1
                candidate_white = assignment.white.kind is PlayerKind.MODEL
                wins = (assignment.context.rung_id or "") in self.beats
                white_wins = wins == candidate_white
                yield GameRecord(
                    game_id=assignment.game_id,
                    white=assignment.white,
                    black=assignment.black,
                    result=GameResult.WHITE_WINS if white_wins else GameResult.BLACK_WINS,
                    termination=Termination.CHECKMATE,
                    context=assignment.context,
                    created_at=StepClock().now(),
                )
        finally:
            self.closed += 1


def _spec() -> LadderSpec:
    rungs = tuple(
        Rung(
            rung_id=name, label=name, engine=EngineSpec(family=EngineFamily.RANDOM), nominal_elo=elo
        )
        for name, elo in (("a", 400), ("b", 800), ("c", 1200))
    )
    return LadderSpec(ladder_id="ladder-v1", version="1", rungs=rungs, games_per_rung=4)


def _service(home: Path, beats: set[str]) -> tuple[LadderService, ScriptedRunner, FileGameStore]:
    models = FileModelStore(home)
    recipe = TrainingRecipe(network=NetworkSpec(hidden=(8,)))
    card = ModelCard(
        model_id="cand",
        label="Candidate",
        recipe=recipe,
        recipe_digest=recipe.digest(),
        created_at=StepClock().now(),
    )
    models.save(card, ValueNetwork.initialize(recipe.network, np.random.default_rng(0)))
    runner, games = ScriptedRunner(beats), FileGameStore(home)
    stores = LadderStores(Ladders(_spec()), models, FileJobStore(home), games)
    return LadderService(stores, runner, StepClock()), runner, games


def test_gauntlet_stops_at_the_first_rung_not_beaten(tmp_path: Path) -> None:
    service, runner, games = _service(tmp_path, {"a", "b"})
    run = service.run(LadderRunRequest(model_id="cand"))
    assert run.status is JobStatus.COMPLETED
    report = run.report
    assert report is not None
    assert report.complete
    assert [result.rung_id for result in report.results] == ["a", "b", "c"]
    assert report.ladder_score == pytest.approx(2.0)
    assert report.highest_beaten == "b"
    assert report.elo_estimate is not None
    assert len(games.recent(None, 100)) == runner.requested
    assert runner.closed == 3


def test_early_decisions_skip_remaining_games(tmp_path: Path) -> None:
    service, runner, _ = _service(tmp_path, {"a"})
    run = service.run(LadderRunRequest(model_id="cand"))
    assert run.report is not None
    first, second = run.report.results
    assert first.passed
    assert first.games == 3
    assert not second.passed
    assert second.games == 2
    assert runner.requested == 5


def test_unknown_model_is_rejected_before_queueing(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path, set())
    with pytest.raises(NotFoundError):
        _ = service.submit(LadderRunRequest(model_id="missing"))
