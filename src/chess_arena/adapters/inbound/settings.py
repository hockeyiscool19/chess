"""Operator settings from ``CHESS_ARENA_*`` environment variables."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_workers() -> int:
    """Leave one core for the web server and the operating system."""
    return max(1, (os.cpu_count() or 2) - 1)


class ArenaSettings(BaseSettings):
    """Where state lives, which Stockfish to use, and how much parallelism to allow."""

    model_config = SettingsConfigDict(env_prefix="CHESS_ARENA_", extra="ignore")

    home: Path = Field(default=Path("var"), description="Games, models, and job records.")
    stockfish: str | None = Field(default=None, description="Stockfish binary; auto-detected.")
    host: str = "127.0.0.1"
    port: int = Field(default=8744, ge=1, le=65535)
    workers: int = Field(default_factory=_default_workers, ge=1, le=256)
    human_move_seconds: float = Field(default=2.0, gt=0, le=60)
