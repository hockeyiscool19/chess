"""Ladder runs and matches: queue, poll, and list."""

from fastapi import APIRouter

from chess_arena.adapters.inbound.api.jobs import BackgroundJobs
from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.domain.jobs import LadderRun, LadderRunRequest, MatchRequest, MatchRun


def jobs_router(arena: Arena, jobs: BackgroundJobs) -> APIRouter:
    """Return routes for background jobs."""
    router = APIRouter(prefix="/api", tags=["jobs"])

    @router.post("/ladder-runs", status_code=202)
    def queue_ladder(request: LadderRunRequest) -> LadderRun:
        """Queue a gauntlet for a stored model; poll the returned run for progress."""
        return jobs.ladder(request)

    @router.get("/ladder-runs")
    def ladder_runs(model_id: str | None = None) -> list[LadderRun]:
        """List ladder runs, newest first."""
        return list(arena.jobs.ladder_runs(model_id))

    @router.get("/ladder-runs/{run_id}")
    def ladder_run(run_id: str) -> LadderRun:
        """Return one ladder run with its (possibly partial) report."""
        return arena.jobs.ladder_run(run_id)

    @router.post("/matches", status_code=202)
    def queue_match(request: MatchRequest) -> MatchRun:
        """Queue a match between two automated players."""
        return jobs.match(request)

    @router.get("/matches/{run_id}")
    def match(run_id: str) -> MatchRun:
        """Return one match."""
        return arena.jobs.match(run_id)

    return router
