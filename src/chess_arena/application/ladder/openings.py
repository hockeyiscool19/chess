"""A small book of balanced openings so ladder games do not repeat.

Each pair of ladder games uses one opening with colors swapped, which cancels
most of any advantage the opening gives one side.
"""

from chess_arena.application.ports.runtime import Opening

OPENINGS: tuple[Opening, ...] = (
    Opening(name="Italian Game", moves=("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5")),
    Opening(name="Queen's Gambit Declined", moves=("d2d4", "d7d5", "c2c4", "e7e6", "b1c3")),
    Opening(name="Sicilian Defence", moves=("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4")),
    Opening(name="French Defence", moves=("e2e4", "e7e6", "d2d4", "d7d5", "b1c3")),
    Opening(name="Caro-Kann Defence", moves=("e2e4", "c7c6", "d2d4", "d7d5", "e4e5")),
    Opening(name="Ruy Lopez", moves=("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6")),
    Opening(name="English Opening", moves=("c2c4", "e7e5", "b1c3", "g8f6", "g1f3")),
    Opening(name="King's Indian Defence", moves=("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7")),
    Opening(name="Slav Defence", moves=("d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6")),
    Opening(name="Scandinavian Defence", moves=("e2e4", "d7d5", "e4d5", "d8d5", "b1c3")),
    Opening(name="Pirc Defence", moves=("e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6")),
    Opening(name="Nimzo-Indian Defence", moves=("d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4")),
    Opening(name="London System", moves=("d2d4", "d7d5", "g1f3", "g8f6", "c1f4")),
    Opening(name="Scotch Game", moves=("e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4")),
    Opening(name="Dutch Defence", moves=("d2d4", "f7f5", "g2g3", "g8f6", "f1g2")),
    Opening(name="Reti Opening", moves=("g1f3", "d7d5", "c2c4", "e7e6", "g2g3")),
)


def opening_for_pair(pair_index: int, offset: int = 0) -> Opening:
    """Return the opening used by game pair ``pair_index`` (both colors share it)."""
    return OPENINGS[(pair_index + offset) % len(OPENINGS)]
