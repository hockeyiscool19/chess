"""System clock."""

from datetime import UTC, datetime


class SystemClock:
    """Aware UTC time from the operating system."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(UTC)
