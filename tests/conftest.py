"""Shared fixtures: a stepping clock and a temporary inline arena."""

from collections.abc import Generator
from pathlib import Path

import pytest

from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.adapters.inbound.settings import ArenaSettings
from tests.helpers import StepClock


@pytest.fixture
def clock() -> StepClock:
    return StepClock()


@pytest.fixture
def arena(tmp_path: Path) -> Generator[Arena]:
    settings = ArenaSettings(home=tmp_path / "arena", workers=1, stockfish="definitely-missing")
    built = Arena(settings, inline=True)
    yield built
    built.close()
