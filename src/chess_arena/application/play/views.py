"""Board views, replays, and PGN text derived from stored game records."""

import chess
import chess.pgn

from chess_arena.domain.games import Color, GameRecord, GameReplay, GameView, ReplayFrame


def _board(record: GameRecord) -> tuple[chess.Board, list[str]]:
    """Replay a record, returning the final board and the SAN of every move."""
    board = chess.Board(record.start_fen)
    sans: list[str] = []
    for uci in record.moves:
        move = chess.Move.from_uci(uci)
        sans.append(board.san(move))
        board.push(move)
    return board, sans


def view_of(record: GameRecord) -> GameView:
    """Return the current position, legal moves, and move list of ``record``."""
    board, sans = _board(record)
    legal = () if record.finished else tuple(move.uci() for move in board.legal_moves)
    king = board.king(board.turn)
    check = chess.square_name(king) if board.is_check() and king is not None else None
    return GameView(
        game=record,
        fen=board.fen(),
        turn=Color.WHITE if board.turn == chess.WHITE else Color.BLACK,
        legal_moves=legal,
        san_moves=tuple(sans),
        last_move=record.moves[-1] if record.moves else None,
        check_square=check,
        ply=len(record.moves),
    )


def pgn_of(record: GameRecord) -> str:
    """Return the game as PGN text with player and result headers."""
    board, _ = _board(record)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"Chess Arena {record.context.kind.value} game"
    game.headers["Site"] = "chess-arena"
    game.headers["Date"] = record.created_at.strftime("%Y.%m.%d")
    game.headers["White"] = record.white.key
    game.headers["Black"] = record.black.key
    game.headers["Result"] = record.result.value
    if record.context.opening:
        game.headers["Opening"] = record.context.opening
    if record.termination is not None:
        game.headers["Termination"] = record.termination.value
    return str(game)


def replay_of(record: GameRecord) -> GameReplay:
    """Return every position of the game for step-by-step viewing."""
    board = chess.Board(record.start_fen)
    frames = [ReplayFrame(ply=0, fen=board.fen())]
    for ply, uci in enumerate(record.moves, start=1):
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        board.push(move)
        frames.append(ReplayFrame(ply=ply, fen=board.fen(), san=san, uci=uci))
    return GameReplay(game=record, frames=tuple(frames), pgn=pgn_of(record))
