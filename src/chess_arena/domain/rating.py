"""Elo arithmetic: expected score and a performance rating against rated opponents."""

from chess_arena.domain.ladder import RungResult

_SCALE = 400.0
_LOW = -1000.0
_HIGH = 5000.0
_ITERATIONS = 80
_MARGIN = 400.0


def expected_score(rating: float, opponent: float) -> float:
    """Return the Elo-model expected score of ``rating`` against ``opponent``."""
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / _SCALE))


def performance_rating(results: tuple[RungResult, ...]) -> float | None:
    """Return the rating whose expected points against the rungs equal the actual points.

    Perfect or zero scores have no finite solution, so they are clamped to 400
    points above the strongest or below the weakest opponent faced.
    """
    played = [result for result in results if result.games]
    if not played:
        return None
    points = sum(result.points for result in played)
    games = sum(result.games for result in played)
    if points >= games:
        return max(result.nominal_elo for result in played) + _MARGIN
    if points <= 0:
        return min(result.nominal_elo for result in played) - _MARGIN
    low, high = _LOW, _HIGH
    for _ in range(_ITERATIONS):
        middle = (low + high) / 2
        expected = sum(
            result.games * expected_score(middle, result.nominal_elo) for result in played
        )
        if expected < points:
            low = middle
        else:
            high = middle
    return round((low + high) / 2, 1)
