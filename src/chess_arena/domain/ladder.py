"""The engine ladder: ordered opponents, gauntlet results, and the ladder score.

A gauntlet plays a candidate against each rung in order. A rung is *beaten* when
the candidate scores at least ``pass_score`` of the points (a win is 1, a draw
0.5). The gauntlet stops at the first rung that is not beaten. The ladder score
is the number of beaten rungs plus the score fraction on the first failed rung,
so ``3.4`` means "beat three rungs and took 40% of the points on the fourth".
"""

import hashlib
import json
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from chess_arena.domain.contract import ArenaModel, ShortText, Slug
from chess_arena.domain.players import EngineSpec, PlayerRef


class Rung(ArenaModel):
    """One opponent on the ladder."""

    rung_id: Slug
    label: ShortText
    engine: EngineSpec
    nominal_elo: int = Field(ge=0, le=4000)
    description: str = ""


class LadderSpec(ArenaModel):
    """A versioned ladder protocol: rungs, games per rung, and the pass rule."""

    ladder_id: Slug
    version: Slug
    rungs: tuple[Rung, ...] = Field(min_length=1)
    games_per_rung: int = Field(default=8, ge=2, le=200)
    pass_score: float = Field(default=0.55, gt=0.5, le=1.0)
    max_plies: int = Field(default=300, ge=20, le=2000)
    candidate_max_seconds_per_move: float = Field(default=1.0, gt=0, le=60)

    @model_validator(mode="after")
    def rungs_are_ordered(self) -> Self:
        """Require unique rung IDs, an even game count, and non-decreasing Elo."""
        ids = [rung.rung_id for rung in self.rungs]
        if len(set(ids)) != len(ids):
            msg = "rung IDs must be unique"
            raise ValueError(msg)
        if self.games_per_rung % 2:
            msg = "games_per_rung must be even so both colors are played equally"
            raise ValueError(msg)
        elos = [rung.nominal_elo for rung in self.rungs]
        if elos != sorted(elos):
            msg = "rungs must be ordered by nominal Elo"
            raise ValueError(msg)
        return self

    def rung(self, rung_id: str) -> Rung:
        """Return the rung with ``rung_id``."""
        for rung in self.rungs:
            if rung.rung_id == rung_id:
                return rung
        msg = f"unknown rung {rung_id}"
        raise KeyError(msg)

    def limited(self, max_rungs: int | None, games_per_rung: int | None) -> "LadderSpec":
        """Return this spec truncated to ``max_rungs`` and with an optional game count."""
        rungs = self.rungs if max_rungs is None else self.rungs[:max_rungs]
        games = self.games_per_rung if games_per_rung is None else games_per_rung
        return self.model_copy(update={"rungs": rungs, "games_per_rung": games})

    def digest(self) -> str:
        """Return a SHA-256 over the canonical JSON of the whole protocol."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class RungResult(ArenaModel):
    """Points a candidate earned against one rung."""

    rung_id: Slug
    nominal_elo: int = Field(ge=0, le=4000)
    games_planned: int = Field(ge=1)
    wins: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    passed: bool = False
    decided: bool = False
    game_ids: tuple[Slug, ...] = ()

    @property
    def games(self) -> int:
        """Return the number of finished games."""
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        """Return wins plus half the draws."""
        return self.wins + 0.5 * self.draws

    @property
    def score(self) -> float:
        """Return the point fraction over finished games (0 before any game)."""
        return self.points / self.games if self.games else 0.0


def rung_decision(result: RungResult, pass_score: float) -> bool | None:
    """Return True (beaten), False (not beaten), or None while still open.

    The decision is final as soon as the remaining games cannot change it.
    """
    needed = pass_score * result.games_planned
    remaining = result.games_planned - result.games
    if result.points >= needed:
        return True
    if result.points + remaining < needed or remaining == 0:
        return False
    return None


def ladder_score(results: tuple[RungResult, ...]) -> float:
    """Return beaten rungs plus the score fraction on the first rung not beaten."""
    total = 0.0
    for result in results:
        if not result.passed:
            return total + result.score
        total += 1.0
    return total


class LadderReport(ArenaModel):
    """The outcome of one gauntlet for one candidate."""

    report_id: Slug
    ladder_id: Slug
    ladder_version: Slug
    spec_digest: str = Field(min_length=64, max_length=64)
    player: PlayerRef
    results: tuple[RungResult, ...] = ()
    ladder_score: float = Field(default=0.0, ge=0)
    highest_beaten: Slug | None = None
    elo_estimate: float | None = None
    complete: bool = False
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
