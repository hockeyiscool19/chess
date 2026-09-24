"""768 piece-square features seen from the side to move.

Each piece sets one of 12 x 64 inputs: its type, whether it belongs to the side
to move ("mine") or the opponent, and its square. When Black is to move the board
is mirrored vertically so the network always evaluates "my" position; one set of
weights therefore serves both colors.
"""

import chess
import numpy as np
from numpy.typing import NDArray

FEATURES = 768
MAX_PIECES = 32
_PLANE = 64
_THEIRS = 6


def feature_indices(board: chess.Board) -> list[int]:
    """Return the active feature indices of ``board`` for the side to move."""
    mover = board.turn
    flip = mover == chess.BLACK
    indices: list[int] = []
    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1 + (0 if piece.color == mover else _THEIRS)
        relative = chess.square_mirror(square) if flip else square
        indices.append(plane * _PLANE + relative)
    return indices


def index_matrix(rows: list[list[int]]) -> NDArray[np.int16]:
    """Pack variable-length index lists into an ``(n, 32)`` array padded with -1."""
    matrix = np.full((len(rows), MAX_PIECES), -1, dtype=np.int16)
    for row, indices in enumerate(rows):
        matrix[row, : len(indices)] = indices
    return matrix


def dense(indices: NDArray[np.int16]) -> NDArray[np.float32]:
    """Expand padded index rows into a dense ``(n, 768)`` 0/1 matrix."""
    count, width = int(indices.shape[0]), int(indices.shape[1])
    batch = np.zeros((count, FEATURES + 1), dtype=np.float32)
    rows = np.repeat(np.arange(count, dtype=np.int64), width)
    columns = indices.reshape(-1).astype(np.int64)
    columns[columns < 0] = FEATURES
    batch[rows, columns] = 1.0
    return batch[:, :FEATURES]
