"""Run ladder runs and matches in background threads that share the worker pool."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.domain.jobs import JobStatus, LadderRun, LadderRunRequest, MatchRequest, MatchRun

_LOG = logging.getLogger(__name__)
_CONCURRENT_JOBS = 4


class BackgroundJobs:
    """Queue jobs; each job's games still run in the shared process pool."""

    def __init__(self, arena: Arena) -> None:
        """Bind the arena and fail any job a previous server left unfinished."""
        self._arena = arena
        self._threads = ThreadPoolExecutor(max_workers=_CONCURRENT_JOBS, thread_name_prefix="job")
        for run in arena.jobs.ladder_runs(None):
            if run.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                arena.jobs.save_ladder_run(
                    run.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "error": "the arena server restarted before this run finished",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )

    def ladder(self, request: LadderRunRequest) -> LadderRun:
        """Queue a ladder run and return it immediately."""
        run = self._arena.ladder.submit(request)
        _ = self._threads.submit(self._execute_ladder, run.run_id)
        return run

    def match(self, request: MatchRequest) -> MatchRun:
        """Queue a match and return it immediately."""
        run = self._arena.matches.submit(request)
        _ = self._threads.submit(self._execute_match, run.run_id)
        return run

    def _execute_ladder(self, run_id: str) -> None:
        """Run a ladder job, logging (the run records) any failure."""
        try:
            _ = self._arena.ladder.execute(run_id)
        except Exception:
            _LOG.exception("ladder run %s failed", run_id)

    def _execute_match(self, run_id: str) -> None:
        """Run a match job, logging (the match records) any failure."""
        try:
            _ = self._arena.matches.execute(run_id)
        except Exception:
            _LOG.exception("match %s failed", run_id)

    def shutdown(self) -> None:
        """Stop accepting jobs."""
        self._threads.shutdown(wait=False, cancel_futures=True)
