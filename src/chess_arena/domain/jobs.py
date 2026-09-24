"""Long-running ladder runs and matches executed by the arena backend."""

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from chess_arena.domain.contract import ArenaModel, Slug
from chess_arena.domain.ladder import LadderReport
from chess_arena.domain.players import PlayerKind, PlayerRef


class JobStatus(StrEnum):
    """Lifecycle of a background job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LadderRunRequest(ArenaModel):
    """Evaluate one model on a ladder, optionally truncated or with fewer games."""

    model_id: Slug
    ladder_id: Slug = "ladder-v1"
    max_rungs: int | None = Field(default=None, ge=1, le=100)
    games_per_rung: int | None = Field(default=None, ge=2, le=200)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class LadderRun(ArenaModel):
    """A queued, running, or finished gauntlet. ``report`` fills in as games finish."""

    run_id: Slug
    request: LadderRunRequest
    status: JobStatus = JobStatus.QUEUED
    report: LadderReport | None = None
    error: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class MatchRequest(ArenaModel):
    """Play ``games`` games between two non-human players, alternating colors."""

    first: PlayerRef
    second: PlayerRef
    games: int = Field(default=2, ge=1, le=200)
    max_plies: int = Field(default=300, ge=20, le=2000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def no_humans(self) -> Self:
        """Matches are automated; humans play through games instead."""
        if PlayerKind.HUMAN in (self.first.kind, self.second.kind):
            msg = "matches cannot seat a human"
            raise ValueError(msg)
        return self


class MatchScore(ArenaModel):
    """Points from the first player's perspective."""

    wins: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)

    @property
    def games(self) -> int:
        """Return the number of finished games."""
        return self.wins + self.draws + self.losses


class MatchRun(ArenaModel):
    """A queued, running, or finished match."""

    run_id: Slug
    request: MatchRequest
    status: JobStatus = JobStatus.QUEUED
    score: MatchScore = MatchScore()
    game_ids: tuple[Slug, ...] = ()
    error: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
