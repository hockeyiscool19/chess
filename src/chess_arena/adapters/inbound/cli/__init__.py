"""``chess-arena`` command-line entry point."""

from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.adapters.inbound.cli import handlers
from chess_arena.adapters.inbound.cli.argv import build_parser
from chess_arena.adapters.inbound.cli.output import expected, fail
from chess_arena.adapters.inbound.settings import ArenaSettings


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one command, and return its exit code."""
    args = build_parser().parse_args(argv)
    command = str(args.command)
    label = f"{command} {args.action}" if getattr(args, "action", None) else command
    as_json = bool(getattr(args, "json", False))
    settings = ArenaSettings()
    if command == "serve":
        from chess_arena.adapters.inbound.api.server import serve  # noqa: PLC0415

        overrides = {"home": args.home, "workers": args.workers}
        chosen = {key: value for key, value in overrides.items() if value is not None}
        return serve(settings.model_copy(update=chosen), host=args.host, port=args.port)
    if command == "recipe":
        try:
            return handlers.recipe(args)
        except Exception as exc:
            if expected(exc):
                return fail(label, exc, as_json=as_json)
            raise
    arena = Arena(settings, inline=command not in {"ladder", "match", "train"})
    try:
        handler = {
            "doctor": handlers.doctor,
            "engines": handlers.engines,
            "ladder": handlers.ladder,
            "models": handlers.models,
            "match": handlers.match,
            "train": handlers.train,
        }[command]
        return handler(arena, args)
    except Exception as exc:
        if expected(exc):
            return fail(label, exc, as_json=as_json)
        raise
    finally:
        arena.close()
