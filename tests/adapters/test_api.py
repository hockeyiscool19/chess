import base64
import time
from collections.abc import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chess_arena.adapters.inbound.api.app import create_app
from chess_arena.adapters.inbound.bootstrap import Arena
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.learning.serialization import to_npz_bytes
from chess_arena.domain.models import ModelCard
from chess_arena.domain.recipes import NetworkSpec, SearchSpec, TrainingRecipe
from tests.helpers import StepClock


@pytest.fixture
def client(arena: Arena) -> Generator[TestClient]:
    with TestClient(create_app(arena)) as test_client:
        yield test_client


def _upload(model_id: str = "net-1") -> dict[str, object]:
    recipe = TrainingRecipe(
        network=NetworkSpec(hidden=(8,)),
        search=SearchSpec(depth=1, max_seconds_per_move=0.1, material_blend=1.0),
    )
    card = ModelCard(
        model_id=model_id,
        label="Net one",
        recipe=recipe,
        recipe_digest=recipe.digest(),
        created_at=StepClock().now(),
    )
    network = ValueNetwork.initialize(recipe.network, np.random.default_rng(0))
    weights = base64.b64encode(to_npz_bytes(network)).decode()
    return {"card": card.model_dump(mode="json"), "weights_npz_base64": weights}


def test_health_and_openapi(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    schema = client.get("/openapi.json").json()
    assert "/api/ladder-runs" in schema["paths"]
    assert client.get("/").status_code == 200


def test_play_a_move_against_a_bot(client: TestClient) -> None:
    started = client.post(
        "/api/games",
        json={"opponent": {"kind": "engine", "player_id": "random"}, "human_color": "white"},
    )
    assert started.status_code == 201
    game_id = started.json()["game"]["game_id"]
    moved = client.post(f"/api/games/{game_id}/moves", json={"uci": "e2e4"})
    assert moved.status_code == 200
    assert moved.json()["ply"] == 2
    illegal = client.post(f"/api/games/{game_id}/moves", json={"uci": "e2e4"})
    assert illegal.status_code == 422
    assert client.get(f"/api/games/{game_id}/replay").json()["frames"][0]["ply"] == 0
    assert client.get("/api/games/missing").status_code == 404


def test_model_upload_and_ladder_run(client: TestClient) -> None:
    assert client.post("/api/models", json=_upload()).status_code == 201
    assert client.post("/api/models", json=_upload()).status_code == 201
    bad = _upload("net-2")
    bad["weights_npz_base64"] = "not-base64!"
    assert client.post("/api/models", json=bad).status_code == 422
    queued = client.post(
        "/api/ladder-runs", json={"model_id": "net-1", "max_rungs": 1, "games_per_rung": 2}
    )
    assert queued.status_code == 202
    run_id = queued.json()["run_id"]
    deadline = time.monotonic() + 60
    run = client.get(f"/api/ladder-runs/{run_id}").json()
    while run["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(0.2)
        run = client.get(f"/api/ladder-runs/{run_id}").json()
    assert run["status"] == "completed", run
    assert run["report"]["results"][0]["rung_id"] == "random"
    models = client.get("/api/models").json()
    assert models[0]["reports"] == 1
    opponents = [item["player"]["player_id"] for item in client.get("/api/opponents").json()]
    assert "net-1" in opponents
