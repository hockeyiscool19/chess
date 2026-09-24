"""Base model configuration and identifier types for every arena contract."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")]
"""Lower-case identifier safe to use as a file name and URL path segment."""

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=200)]
"""A short non-empty human-readable label."""

LongText = Annotated[str, StringConstraints(max_length=8000)]
"""Free text such as notes or a proposal summary; may be empty."""

UciMove = Annotated[str, StringConstraints(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$")]
"""A move in UCI long algebraic notation, for example ``e2e4`` or ``e7e8q``."""

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class ArenaModel(BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")
