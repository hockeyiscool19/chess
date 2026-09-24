import pytest
from pydantic import ValidationError

from chess_arena.domain.ladder import LadderSpec, Rung, RungResult, ladder_score, rung_decision
from chess_arena.domain.players import EngineFamily, EngineSpec, PlayerKind, PlayerRef
from chess_arena.domain.rating import expected_score, performance_rating


def _rung(rung_id: str, elo: int) -> Rung:
    return Rung(
        rung_id=rung_id,
        label=rung_id,
        engine=EngineSpec(family=EngineFamily.RANDOM),
        nominal_elo=elo,
    )


def _result(
    rung_id: str, record: tuple[int, int, int], *, passed: bool, elo: int = 1000
) -> RungResult:
    wins, draws, losses = record
    return RungResult(
        rung_id=rung_id,
        nominal_elo=elo,
        games_planned=8,
        wins=wins,
        draws=draws,
        losses=losses,
        passed=passed,
    )


def test_rung_decision_passes_as_soon_as_points_suffice() -> None:
    result = RungResult(rung_id="r", nominal_elo=0, games_planned=8, wins=5)
    assert rung_decision(result, 0.55) is True


def test_rung_decision_fails_when_remaining_games_cannot_help() -> None:
    result = RungResult(rung_id="r", nominal_elo=0, games_planned=8, wins=1, losses=4)
    assert rung_decision(result, 0.55) is False


def test_rung_decision_stays_open_while_undecided() -> None:
    result = RungResult(rung_id="r", nominal_elo=0, games_planned=8, wins=2, losses=2)
    assert rung_decision(result, 0.55) is None


def test_ladder_score_counts_beaten_rungs_plus_partial_credit() -> None:
    results = (
        _result("a", (5, 0, 0), passed=True),
        _result("b", (5, 1, 0), passed=True),
        _result("c", (1, 2, 3), passed=False),
    )
    assert ladder_score(results) == pytest.approx(2 + 2 / 6)


def test_ladder_score_of_a_full_climb_is_the_rung_count() -> None:
    results = (_result("a", (5, 0, 0), passed=True), _result("b", (5, 0, 0), passed=True))
    assert ladder_score(results) == 2.0


def test_spec_rejects_unordered_or_duplicate_rungs() -> None:
    with pytest.raises(ValidationError):
        _ = LadderSpec(ladder_id="l", version="1", rungs=(_rung("a", 500), _rung("b", 400)))
    with pytest.raises(ValidationError):
        _ = LadderSpec(ladder_id="l", version="1", rungs=(_rung("a", 400), _rung("a", 500)))
    with pytest.raises(ValidationError):
        _ = LadderSpec(ladder_id="l", version="1", rungs=(_rung("a", 400),), games_per_rung=7)


def test_limited_spec_changes_the_digest() -> None:
    spec = LadderSpec(ladder_id="l", version="1", rungs=(_rung("a", 400), _rung("b", 500)))
    limited = spec.limited(1, 4)
    assert [rung.rung_id for rung in limited.rungs] == ["a"]
    assert limited.games_per_rung == 4
    assert limited.digest() != spec.digest()


def test_performance_rating_matches_even_score_and_clamps_extremes() -> None:
    even = performance_rating((_result("a", (2, 0, 2), passed=False, elo=1500),))
    assert even == pytest.approx(1500, abs=1)
    perfect = performance_rating((_result("a", (4, 0, 0), passed=True, elo=1200),))
    assert perfect == 1600
    assert performance_rating(()) is None
    assert expected_score(1600, 1200) > 0.9


def test_player_ref_round_trips_through_its_key() -> None:
    ref = PlayerRef(kind=PlayerKind.ENGINE, player_id="stockfish-1320")
    assert PlayerRef.parse(ref.key) == ref


def test_engine_spec_parameters_must_match_family() -> None:
    with pytest.raises(ValidationError):
        _ = EngineSpec(family=EngineFamily.MINIMAX)
    with pytest.raises(ValidationError):
        _ = EngineSpec(family=EngineFamily.RANDOM, elo=1500)
    assert EngineSpec(family=EngineFamily.STOCKFISH, elo=1500).elo == 1500
