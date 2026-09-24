"""Hand-written evaluation: material plus simple piece-square terms, tapered by phase.

Scores are centipawns from the side to move's point of view (negamax convention).
The tables are generated from file and rank formulas rather than copied from an
engine, so every number here is explainable in one line.
"""

import chess

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
_PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
_FULL_PHASE = 24
_BISHOP_PAIR = 30
_MOP_UP_MARGIN = 300
_MOP_UP_PHASE = 8
_SEVENTH_RANK = 6
_CASTLED_FILES = frozenset({1, 2, 6})


def _centrality(square: chess.Square) -> int:
    """Return 0 on the edge ring up to 3 on the four central squares."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    return 3 - int(max(abs(file - 3.5), abs(rank - 3.5)))


def _pawn_bonus(square: chess.Square) -> int:
    """Reward advanced pawns and central pawns on ranks 3 to 5 (white's view)."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    advance = 4 * (rank - 1) + (12 if rank == _SEVENTH_RANK else 0)
    central = 12 if file in (3, 4) and 2 <= rank <= 4 else 0  # noqa: PLR2004  # ranks 3-5
    return advance + central


def _rook_bonus(square: chess.Square) -> int:
    """Reward rooks on the seventh rank and on central files."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    return (15 if rank == _SEVENTH_RANK else 0) + (5 if file in (3, 4) else 0)


def _king_middlegame(square: chess.Square) -> int:
    """Reward a castled king on the back rank; punish a wandering king."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    if rank == 0:
        return 20 if file in _CASTLED_FILES else 0
    return -20 * min(rank, 3)


def _king_endgame(square: chess.Square) -> int:
    """Reward a centralized king once the heavy pieces are gone."""
    return 12 * _centrality(square) - 18


MIDDLEGAME: dict[chess.PieceType, tuple[int, ...]] = {
    chess.PAWN: tuple(_pawn_bonus(sq) for sq in chess.SQUARES),
    chess.KNIGHT: tuple(-40 + 20 * _centrality(sq) for sq in chess.SQUARES),
    chess.BISHOP: tuple(-10 + 8 * _centrality(sq) for sq in chess.SQUARES),
    chess.ROOK: tuple(_rook_bonus(sq) for sq in chess.SQUARES),
    chess.QUEEN: tuple(-5 + 4 * _centrality(sq) for sq in chess.SQUARES),
    chess.KING: tuple(_king_middlegame(sq) for sq in chess.SQUARES),
}
ENDGAME: dict[chess.PieceType, tuple[int, ...]] = {
    **MIDDLEGAME,
    chess.PAWN: tuple(2 * _pawn_bonus(sq) for sq in chess.SQUARES),
    chess.KING: tuple(_king_endgame(sq) for sq in chess.SQUARES),
}


def game_phase(board: chess.Board) -> int:
    """Return 24 with all minor and major pieces on the board, falling to 0."""
    phase = sum(
        weight * len(board.pieces(piece_type, color))
        for piece_type, weight in _PHASE_WEIGHTS.items()
        for color in chess.COLORS
    )
    return min(phase, _FULL_PHASE)


def white_score(board: chess.Board) -> int:
    """Return the static evaluation in centipawns from White's point of view."""
    middle = 0
    end = 0
    for square, piece in board.piece_map().items():
        relative = square if piece.color == chess.WHITE else chess.square_mirror(square)
        value = PIECE_VALUES[piece.piece_type]
        sign = 1 if piece.color == chess.WHITE else -1
        middle += sign * (value + MIDDLEGAME[piece.piece_type][relative])
        end += sign * (value + ENDGAME[piece.piece_type][relative])
    phase = game_phase(board)
    score = (middle * phase + end * (_FULL_PHASE - phase)) // _FULL_PHASE
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if len(board.pieces(chess.BISHOP, color)) >= 2:  # noqa: PLR2004  # a pair
            score += sign * _BISHOP_PAIR
    return score


def material_balance(board: chess.Board) -> int:
    """Return White's material minus Black's, in centipawns."""
    return sum(
        (1 if piece.color == chess.WHITE else -1) * PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
    )


def mop_up(board: chess.Board) -> float:
    """Return an endgame bonus (White's view) for driving a lone king to the edge.

    Without it a shallow search that is a rook up often shuffles until the
    fifty-move rule. The bonus applies only when the defender has no pawns and is
    at least a minor piece down, and it is the same for every evaluator.
    """
    balance = material_balance(board)
    if abs(balance) < _MOP_UP_MARGIN or game_phase(board) > _MOP_UP_PHASE:
        return 0.0
    strong = chess.WHITE if balance > 0 else chess.BLACK
    if board.pieces(chess.PAWN, not strong):
        return 0.0
    weak_king = board.king(not strong)
    strong_king = board.king(strong)
    if weak_king is None or strong_king is None:
        return 0.0
    edge = 3 - _centrality(weak_king)
    closeness = 14 - chess.square_manhattan_distance(weak_king, strong_king)
    bonus = 10.0 * edge + 4.0 * closeness
    return bonus if strong == chess.WHITE else -bonus


def evaluate(board: chess.Board) -> float:
    """Return the static evaluation from the side to move's point of view."""
    score = white_score(board) + mop_up(board)
    return float(score if board.turn == chess.WHITE else -score)


class MaterialEvaluator:
    """Evaluator protocol adapter around :func:`evaluate`."""

    def evaluate(self, board: chess.Board) -> float:
        """Return centipawns for the side to move."""
        return evaluate(board)
