"""Model listings that combine a card with its best ladder result."""

from chess_arena.domain.ladder import LadderReport
from chess_arena.domain.models import ModelCard, ModelSummary


def summarize(card: ModelCard, reports: tuple[LadderReport, ...]) -> ModelSummary:
    """Return ``card`` with the best ladder score and Elo estimate among ``reports``."""
    complete = [report for report in reports if report.complete]
    if not complete:
        return ModelSummary(card=card, reports=len(reports))
    best = max(complete, key=lambda report: report.ladder_score)
    return ModelSummary(
        card=card,
        best_ladder_score=best.ladder_score,
        best_elo_estimate=best.elo_estimate,
        highest_beaten=best.highest_beaten,
        reports=len(reports),
    )
