"""Fit a value network to (position, target) pairs with early stopping."""

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from chess_arena.application.learning.features import dense
from chess_arena.application.learning.network import Array, ValueNetwork
from chess_arena.application.learning.optimizer import Adam
from chess_arena.domain.contract import ArenaModel
from chess_arena.domain.recipes import OptimizerSpec

_MIN_VALIDATION = 32


class FitReport(ArenaModel):
    """Losses and the number of epochs actually run."""

    epochs_run: int = Field(ge=0)
    train_loss: float | None = None
    validation_loss: float | None = None
    stopped_early: bool = False


def _mse(network: ValueNetwork, indices: NDArray[np.int16], targets: Array, batch: int) -> float:
    """Return the mean squared error over a dataset, evaluated in batches."""
    total = 0.0
    for start in range(0, len(targets), batch):
        output, _ = network.forward(dense(indices[start : start + batch]))
        total += float(((output - targets[start : start + batch]) ** 2).sum())
    return total / max(1, len(targets))


def fit(
    network: ValueNetwork,
    indices: NDArray[np.int16],
    targets: Array,
    spec: OptimizerSpec,
    rng: np.random.Generator,
) -> FitReport:
    """Train ``network`` in place and restore the weights with the best validation loss."""
    count = len(targets)
    if count == 0:
        return FitReport(epochs_run=0)
    order = rng.permutation(count)
    held = int(count * spec.validation_fraction)
    if held < _MIN_VALIDATION:
        held = 0
    validation, training = order[:held], order[held:]
    optimizer = Adam(network.parameters(), spec.learning_rate, spec.weight_decay)
    best_loss = float("inf")
    best = network.copy()
    stale = 0
    epochs = 0
    train_loss = 0.0
    for _ in range(spec.epochs):
        epochs += 1
        shuffled = rng.permutation(training)
        running = 0.0
        for start in range(0, len(shuffled), spec.batch_size):
            rows = shuffled[start : start + spec.batch_size]
            output, cache = network.forward(dense(indices[rows]))
            error = output - targets[rows]
            running += float((error**2).sum())
            _ = optimizer.step(
                network.backward(cache, (2.0 * error / len(rows)).astype(np.float32))
            )
        train_loss = running / max(1, len(training))
        loss = _mse(network, indices[validation], targets[validation], 2048) if held else train_loss
        if loss < best_loss - 1e-6:
            best_loss = loss
            best = network.copy()
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    network.weights, network.biases = best.weights, best.biases
    return FitReport(
        epochs_run=epochs,
        train_loss=round(train_loss, 6),
        validation_loss=round(best_loss, 6) if held else None,
        stopped_early=epochs < spec.epochs,
    )
