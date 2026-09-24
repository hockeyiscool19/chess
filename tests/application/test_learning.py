import chess
import numpy as np
import pytest

from chess_arena.application.learning.features import (
    FEATURES,
    dense,
    feature_indices,
    index_matrix,
)
from chess_arena.application.learning.fit import fit
from chess_arena.application.learning.network import ValueNetwork
from chess_arena.application.learning.serialization import from_blob, to_blob
from chess_arena.domain.recipes import Activation, NetworkSpec, OptimizerSpec


def test_features_cover_every_piece_and_are_color_symmetric() -> None:
    board = chess.Board()
    indices = feature_indices(board)
    assert len(indices) == 32
    assert all(0 <= index < FEATURES for index in indices)
    _ = board.push_uci("e2e4")
    mirrored = board.mirror()
    assert sorted(feature_indices(board)) == sorted(feature_indices(mirrored))


def test_dense_expands_padded_rows() -> None:
    matrix = index_matrix([[0, 5], [767]])
    batch = dense(matrix)
    assert batch.shape == (2, FEATURES)
    assert batch[0, 0] == 1.0
    assert batch[0, 5] == 1.0
    assert batch[0].sum() == 2.0
    assert batch[1, 767] == 1.0
    assert batch[1].sum() == 1.0


@pytest.mark.parametrize("activation", [Activation.CRELU, Activation.TANH])
def test_backward_matches_numeric_gradients(activation: Activation) -> None:
    rng = np.random.default_rng(1)
    network = ValueNetwork.initialize(NetworkSpec(hidden=(8, 8), activation=activation), rng)
    for weight in network.weights:
        _ = np.add(weight, rng.normal(0, 0.3, weight.shape).astype(np.float32), out=weight)
    for bias in network.biases:
        _ = np.add(bias, rng.uniform(0.1, 0.4, bias.shape).astype(np.float32), out=bias)
    rows = index_matrix([feature_indices(chess.Board()), [3, 70, 200, 511]])
    inputs = dense(rows).astype(np.float64).astype(np.float32)
    targets = np.array([0.3, -0.2], dtype=np.float32)

    def loss() -> float:
        output, _ = network.forward(inputs)
        return float(((output - targets) ** 2).sum())

    output, cache = network.forward(inputs)
    grads = network.backward(cache, (2.0 * (output - targets)).astype(np.float32))
    for parameter, grad in zip(network.parameters(), grads, strict=True):
        flat = parameter.reshape(-1)
        position = int(np.argmax(np.abs(grad.reshape(-1))))
        original = float(flat[position])
        step = 1e-2
        flat[position] = original + step
        upper = loss()
        flat[position] = original - step
        lower = loss()
        flat[position] = original
        numeric = (upper - lower) / (2 * step)
        assert grad.reshape(-1)[position] == pytest.approx(numeric, rel=0.05, abs=1e-3)


def test_value_of_one_position_matches_the_batch_forward() -> None:
    network = ValueNetwork.initialize(NetworkSpec(hidden=(32, 8)), np.random.default_rng(2))
    network.weights[-1] += 0.5
    indices = feature_indices(chess.Board())
    batch, _ = network.forward(dense(index_matrix([indices])))
    assert network.value(indices) == pytest.approx(float(batch[0]), abs=1e-5)


def test_fit_learns_material_from_labels() -> None:
    rng = np.random.default_rng(4)
    positions: list[list[int]] = []
    targets: list[float] = []
    for index in range(400):
        board = chess.Board()
        up = index % 2 == 0
        victim = chess.square(int(rng.integers(8)), 6 if up else 1)
        _ = board.remove_piece_at(victim)
        positions.append(feature_indices(board))
        targets.append(0.6 if up else -0.6)
    network = ValueNetwork.initialize(NetworkSpec(hidden=(16,)), rng)
    report = fit(
        network,
        index_matrix(positions),
        np.asarray(targets, dtype=np.float32),
        OptimizerSpec(epochs=60, learning_rate=0.01, batch_size=64, patience=50),
        rng,
    )
    assert report.epochs_run > 0
    assert report.validation_loss is not None
    assert report.validation_loss < 0.05


def test_blob_round_trip_preserves_weights() -> None:
    network = ValueNetwork.initialize(NetworkSpec(hidden=(8,)), np.random.default_rng(5))
    copy = from_blob(to_blob(network))
    assert copy.spec == network.spec
    for left, right in zip(network.parameters(), copy.parameters(), strict=True):
        np.testing.assert_array_equal(left, right)
