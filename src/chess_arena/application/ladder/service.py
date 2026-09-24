"""Run a model up the ladder, rung by rung, stopping at the first rung it cannot beat.

Games for one rung run in parallel through the game runner. As soon as the
remaining games cannot change the rung's verdict the rest are cancelled, which
keeps evaluating strong models against weak rungs (and weak models against
strong rungs) cheap. Progress is saved after every game so the web UI and the
institute can watch a run in flight.
"""

import secrets

from chess_arena.application.ladder.openings import opening_for_pair
from chess_arena.application.ports.runtime import ClockPort, GameAssignment, GameRunnerPort
from chess_arena.application.ports.stores import (
    GameStorePort,
    JobStorePort,
    LadderCatalogPort,
    ModelStorePort,
)
from chess_arena.domain.games import Color, GameContext, GameKind, GameRecord
from chess_arena.domain.jobs import JobStatus, LadderRun, LadderRunRequest
from chess_arena.domain.ladder import (
    LadderReport,
    LadderSpec,
    Rung,
    RungResult,
    ladder_score,
    rung_decision,
)
from chess_arena.domain.players import PlayerKind, PlayerRef
from chess_arena.domain.rating import performance_rating


class LadderStores:
    """Persistence ports the ladder service reads and writes."""

    def __init__(
        self,
        catalog: LadderCatalogPort,
        models: ModelStorePort,
        jobs: JobStorePort,
        games: GameStorePort,
    ) -> None:
        """Bind the ladder catalog and the model, job, and game stores."""
        self.catalog = catalog
        self.models = models
        self.jobs = jobs
        self.games = games


class LadderService:
    """Submit and execute ladder runs."""

    def __init__(self, stores: LadderStores, runner: GameRunnerPort, clock: ClockPort) -> None:
        """Bind stores, the parallel game runner, and the clock."""
        self._stores = stores
        self._runner = runner
        self._clock = clock

    def submit(self, request: LadderRunRequest) -> LadderRun:
        """Validate the model and ladder, then store a queued run."""
        _ = self._stores.models.card(request.model_id)
        _ = self._stores.catalog.ladder(request.ladder_id)
        now = self._clock.now()
        run = LadderRun(
            run_id=f"lr-{now:%Y%m%dt%H%M%S}-{secrets.token_hex(3)}",
            request=request,
            created_at=now,
            updated_at=now,
        )
        self._stores.jobs.save_ladder_run(run)
        return run

    def execute(self, run_id: str) -> LadderRun:
        """Play the gauntlet for a queued run and return the finished run."""
        run = self._stores.jobs.ladder_run(run_id)
        request = run.request
        spec = self._stores.catalog.ladder(request.ladder_id).limited(
            request.max_rungs, request.games_per_rung
        )
        started = self._clock.now()
        report = LadderReport(
            report_id=run.run_id,
            ladder_id=spec.ladder_id,
            ladder_version=spec.version,
            spec_digest=spec.digest(),
            player=PlayerRef(kind=PlayerKind.MODEL, player_id=request.model_id),
            started_at=started,
        )
        run = self._save(run, status=JobStatus.RUNNING, report=report)
        try:
            for index, rung in enumerate(spec.rungs):
                result = self._play_rung(run, spec, index, rung)
                report = self._with_result(report, result)
                run = self._save(run, report=report)
                if not result.passed:
                    break
        except Exception as exc:
            _ = self._save(run, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise
        report = report.model_copy(update={"complete": True, "finished_at": self._clock.now()})
        return self._save(run, status=JobStatus.COMPLETED, report=report)

    def run(self, request: LadderRunRequest) -> LadderRun:
        """Submit and execute synchronously (used by the CLI)."""
        return self.execute(self.submit(request).run_id)

    def _play_rung(self, run: LadderRun, spec: LadderSpec, index: int, rung: Rung) -> RungResult:
        """Play one rung's games until the verdict is settled."""
        request = run.request
        candidate = PlayerRef(kind=PlayerKind.MODEL, player_id=request.model_id)
        opponent = PlayerRef(kind=PlayerKind.ENGINE, player_id=rung.rung_id)
        assignments = tuple(
            GameAssignment(
                game_id=f"{run.run_id}-{rung.rung_id}-g{game:02d}",
                white=candidate if game % 2 == 0 else opponent,
                black=opponent if game % 2 == 0 else candidate,
                opening=opening_for_pair(game // 2, offset=request.seed + index),
                max_plies=spec.max_plies,
                seed=request.seed * 100_003 + index * 1_009 + game,
                model_max_seconds=spec.candidate_max_seconds_per_move,
                context=GameContext(
                    kind=GameKind.LADDER,
                    run_id=run.run_id,
                    rung_id=rung.rung_id,
                    opening=opening_for_pair(game // 2, offset=request.seed + index).name,
                ),
            )
            for game in range(spec.games_per_rung)
        )
        result = RungResult(
            rung_id=rung.rung_id, nominal_elo=rung.nominal_elo, games_planned=spec.games_per_rung
        )
        games = self._runner.play(assignments)
        try:
            for game in games:
                self._stores.games.save(game)
                result = _scored(result, game, candidate)
                verdict = rung_decision(result, spec.pass_score)
                if verdict is not None:
                    return result.model_copy(update={"passed": verdict, "decided": True})
        finally:
            games.close()
        verdict = rung_decision(result, spec.pass_score)
        return result.model_copy(update={"passed": bool(verdict), "decided": True})

    def _with_result(self, report: LadderReport, result: RungResult) -> LadderReport:
        """Append a rung result and refresh the score and the Elo estimate."""
        results = (*report.results, result)
        beaten = [item.rung_id for item in results if item.passed]
        return report.model_copy(
            update={
                "results": results,
                "ladder_score": round(ladder_score(results), 4),
                "highest_beaten": beaten[-1] if beaten else None,
                "elo_estimate": performance_rating(results),
            }
        )

    def _save(self, run: LadderRun, **updates: object) -> LadderRun:
        """Persist ``run`` with ``updates`` and a fresh timestamp."""
        saved = run.model_copy(update={**updates, "updated_at": self._clock.now()})
        self._stores.jobs.save_ladder_run(saved)
        return saved


def _scored(result: RungResult, game: GameRecord, candidate: PlayerRef) -> RungResult:
    """Add one finished game to the candidate's tally."""
    color = Color.WHITE if game.white == candidate else Color.BLACK
    points = game.result.points_for(color)
    won, lost = points == 1.0, points == 0.0
    return result.model_copy(
        update={
            "wins": result.wins + int(won),
            "draws": result.draws + int(not won and not lost),
            "losses": result.losses + int(lost),
            "game_ids": (*result.game_ids, game.game_id),
        }
    )
