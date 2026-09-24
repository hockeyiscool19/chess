"""Ladder protocols shipped with the package as JSON files."""

from importlib import resources

from chess_arena.domain.errors import NotFoundError
from chess_arena.domain.ladder import LadderSpec


class PackagedLadders:
    """Load ``*.json`` ladder specs bundled next to this module."""

    def __init__(self) -> None:
        """Parse every bundled ladder once."""
        folder = resources.files(__package__ or "chess_arena.adapters.outbound.ladders")
        specs: dict[str, LadderSpec] = {}
        for entry in folder.iterdir():
            if entry.name.endswith(".json"):
                spec = LadderSpec.model_validate_json(entry.read_text(encoding="utf-8"))
                specs[spec.ladder_id] = spec
        self._specs = specs

    def ladder(self, ladder_id: str) -> LadderSpec:
        """Return one ladder or raise ``NotFoundError``."""
        spec = self._specs.get(ladder_id)
        if spec is None:
            msg = f"unknown ladder {ladder_id}"
            raise NotFoundError(msg)
        return spec

    def ladders(self) -> tuple[LadderSpec, ...]:
        """Return every bundled ladder."""
        return tuple(self._specs[key] for key in sorted(self._specs))
