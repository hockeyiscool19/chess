import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from chess_arena.adapters.inbound.cli import main
from chess_arena.adapters.inbound.cli.output import SCHEMA, Envelope


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "arena"
    monkeypatch.setenv("CHESS_ARENA_HOME", str(home))
    monkeypatch.setenv("CHESS_ARENA_WORKERS", "1")
    monkeypatch.setenv("CHESS_ARENA_STOCKFISH", "definitely-missing-binary")
    return home


def _envelope(capsys: pytest.CaptureFixture[str]) -> Envelope:
    return Envelope.model_validate_json(capsys.readouterr().out.strip().splitlines()[-1])


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def test_recipe_baseline_and_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["recipe", "baseline", "--json"]) == 0
    envelope = _envelope(capsys)
    assert envelope.schema_version == SCHEMA
    assert envelope.ok
    recipe_file = tmp_path / "baseline.json"
    _ = recipe_file.write_text(json.dumps(envelope.data))
    assert main(["recipe", "validate", str(recipe_file), "--json"]) == 0
    digest = _object(_envelope(capsys).data)["digest"]
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_invalid_recipe_is_exit_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe_file = tmp_path / "bad.json"
    _ = recipe_file.write_text('{"search": {"depth": 99}}')
    assert main(["recipe", "validate", str(recipe_file), "--json"]) == 2
    assert not _envelope(capsys).ok


def test_train_register_ladder_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["recipe", "baseline", "--json"]) == 0
    recipe_file = tmp_path / "baseline.json"
    _ = recipe_file.write_text(json.dumps(_envelope(capsys).data))
    out = tmp_path / "models" / "base"
    train = ["train", "--recipe", str(recipe_file), "--out", str(out), "--model-id", "base"]
    assert main([*train, "--register", "--json"]) == 0
    assert _object(_envelope(capsys).data)["model_id"] == "base"
    ladder = ["ladder", "run", "--model", "base", "--max-rungs", "1", "--games-per-rung", "2"]
    assert main([*ladder, "--json"]) == 0
    assert _object(_envelope(capsys).data)["status"] == "completed"
    assert main(["models", "list", "--json"]) == 0
    first = _object(_list(_envelope(capsys).data)[0])
    assert first["reports"] == 1
    assert main([*train, "--json"]) == 1


def test_engines_lists_rungs_without_stockfish(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["engines", "--json"]) == 0
    rows = [_object(row) for row in _list(_envelope(capsys).data)]
    available = {str(_object(row["player"])["player_id"]): row["available"] for row in rows}
    assert available["random"] is True
    assert available["stockfish-full"] is False
