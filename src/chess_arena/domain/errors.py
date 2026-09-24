"""Arena errors mapped to HTTP statuses and CLI exit codes by inbound adapters."""


class ArenaError(Exception):
    """Base class for expected arena failures."""


class NotFoundError(ArenaError):
    """A game, model, rung, or job does not exist."""


class IllegalMoveError(ArenaError):
    """A submitted move is malformed or illegal in the current position."""


class GameOverError(ArenaError):
    """A move or resignation was submitted after the game ended."""


class InvalidRecipeError(ArenaError):
    """A training recipe is inconsistent with its parent model or limits."""


class EngineUnavailableError(ArenaError):
    """An engine binary (for example Stockfish) is not installed or failed to start."""


class ConflictError(ArenaError):
    """A resource with the same identity already exists with different content."""
