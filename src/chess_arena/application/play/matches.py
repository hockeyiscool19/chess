"""Matches between two automated players, alternating colors over book openings."""

import secrets

from chess_arena.application.ladder.openings import opening_for_pair
from chess_arena.application.ports.runtime import ClockPort, GameAssignment, GameRunnerPort
from chess_arena.application.ports.stores import GameStorePort, JobStorePort
from chess_arena.domain.games import Color, GameContext, GameKind
from chess_arena.domain.jobs import JobStatus, MatchRequest, MatchRun, MatchScore


class MatchService:
    """Submit and execute matches."""

    def __init__(
        self, jobs: JobStorePort, games: GameStorePort, runner: GameRunnerPort, clock: ClockPort
    ) -> None:
        """Bind the job and game stores, the game runner, and the clock."""
        self._jobs = jobs
        self._games = games
        self._runner = runner
        self._clock = clock

    def submit(self, request: MatchRequest) -> MatchRun:
        """Store a queued match."""
        now = self._clock.now()
        run = MatchRun(
            run_id=f"mt-{now:%Y%m%dt%H%M%S}-{secrets.token_hex(3)}",
            request=request,
            created_at=now,
            updated_at=now,
        )
        self._jobs.save_match(run)
        return run

    def execute(self, run_id: str) -> MatchRun:
        """Play every game of a queued match."""
        run = self._jobs.match(run_id)
        request = run.request
        run = self._save(run, status=JobStatus.RUNNING)
        assignments = tuple(
            GameAssignment(
                game_id=f"{run.run_id}-g{game:03d}",
                white=request.first if game % 2 == 0 else request.second,
                black=request.second if game % 2 == 0 else request.first,
                opening=opening_for_pair(game // 2, offset=request.seed),
                max_plies=request.max_plies,
                seed=request.seed * 100_003 + game,
                context=GameContext(
                    kind=GameKind.MATCH,
                    run_id=run.run_id,
                    opening=opening_for_pair(game // 2, offset=request.seed).name,
                ),
            )
            for game in range(request.games)
        )
        score = MatchScore()
        try:
            for record in self._runner.play(assignments):
                self._games.save(record)
                color = Color.WHITE if record.white == request.first else Color.BLACK
                points = record.result.points_for(color)
                score = MatchScore(
                    wins=score.wins + int(points == 1.0),
                    draws=score.draws + int(0.0 < points < 1.0),
                    losses=score.losses + int(points == 0.0),
                )
                run = self._save(run, score=score, game_ids=(*run.game_ids, record.game_id))
        except Exception as exc:
            _ = self._save(run, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise
        return self._save(run, status=JobStatus.COMPLETED)

    def run(self, request: MatchRequest) -> MatchRun:
        """Submit and execute synchronously (used by the CLI)."""
        return self.execute(self.submit(request).run_id)

    def _save(self, run: MatchRun, **updates: object) -> MatchRun:
        """Persist ``run`` with ``updates`` and a fresh timestamp."""
        saved = run.model_copy(update={**updates, "updated_at": self._clock.now()})
        self._jobs.save_match(saved)
        return saved
